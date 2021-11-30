import torch
import torch.nn.functional as F
import mmcv

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.models.dense_heads.condlanenet_head import CondLaneHead


class CondLaneNetONNX(torch.nn.Module):
    def __init__(self, model):
        super(CondLaneNetONNX, self).__init__()
        # Layers
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.bbox_head
        # Params
        self.num_classes = 1
        self.num_mask_params = 67
        self.num_gen_params = 134

    def forward(self, img, num_ins=4):
        # # FIXME: inputs.at(2).is_weights() && "The bias tensor is required to be an initializer for the Conv operator
        # w = torch.ones(4, 3, 1, 1, device='cuda:0')
        # # b = torch.ones(4, device='cuda:0')
        # out = F.conv2d(img, w, bias=bias, stride=1, padding=0, groups=1)
        # return out

        # out[0]: 64,  stride = 4
        # out[1]: 128, stride = 8
        # out[2]: 256, stride = 16
        # out[3]: 512, stride = 32
        out = self.backbone(img)
        # out[0]: (1, 64, 40, 100)
        # out[1]: (1, 64, 20, 50)
        # out[2]: (1, 64, 10, 25)
        out, _ = self.neck(out)
        # seeds, hm = self.head.forward_test(out, None, 0.5)

        h_mask, w_mask = out[0].size()[2:]  # 40, 100
        h_hm, w_hm = out[1].size()[2:]      # 20, 50

        z = self.head.ctnet_head(out[1])
        hm, params = z['hm'], z['params']  # (1, 1, 20, 50), (1, 134, 20, 50)
        hm = torch.clamp(hm.sigmoid(), min=1e-4, max=1 - 1e-4)
        # for mask and reg head
        params = params.view(1, 1, -1, h_hm, w_hm)                                          # (1, 1, 134, 20, 50)
        params = params.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.num_gen_params)   # (1000, 134)

        # seeds = self.head.ctdet_decode(hm, thr=hm_thr)
        hmax = F.max_pool2d(hm, 3, stride=1, padding=1)
        keep = (hmax == hm).float()
        heat_nms = hm * keep            # (1, 1, 20, 50)
        heat_nms = heat_nms.flatten()
        # inds = torch.where(heat_nms > hm_thr)
        scores, inds = torch.topk(heat_nms, k=num_ins)

        # parse_pos
        pos_tensor = inds.unsqueeze(1)                              # (num_ins, 1)
        pos_tensor = pos_tensor.expand(-1, self.num_gen_params)     # (num_ins, 134)
        mask_pos_tensor = pos_tensor[:, :self.num_mask_params]      # (num_ins, 67)
        reg_pos_tensor = pos_tensor[:, self.num_mask_params:]       # (num_ins, 67)

        mask_branch = self.head.mask_branch(out[0])
        reg_branch = mask_branch
        mask_params = params[:, :self.num_mask_params].gather(0, mask_pos_tensor)   # (num_ins, 67)
        masks = self.head.mask_head(mask_branch, mask_params, [num_ins])            # (1, num_ins, 40, 100)
        reg_params = params[:, self.num_mask_params:].gather(0, reg_pos_tensor)     # (num_ins, 67)
        regs = self.head.reg_head(reg_branch, reg_params, [num_ins])                # (1, num_ins, 40, 100)
        feat_range = masks.permute(0, 1, 3, 2).view(num_ins, w_mask, h_mask)        # (num_ins, 100, 40)
        feat_range = self.head.mlp(feat_range)                                      # (num_ins, 2, 40)

        return scores, inds, regs[0], masks[0], feat_range


def export_onnx(onnx_file_path):
    # E.g. culane_small
    model_file_path = 'culane_small.pth'
    config_file_path = 'configs/condlanenet/culane/culane_small_test.py'

    # Load specified checkpoint
    cfg = mmcv.Config.fromfile(config_file_path)
    model = build_detector(cfg.model)
    load_checkpoint(model, model_file_path, map_location='cpu')
    model.eval()

    # Export to ONNX
    condlanenet = CondLaneNetONNX(model)
    dummy_input = torch.randn(1, 3, 320, 800)
    torch.onnx.export(condlanenet, dummy_input, onnx_file_path, opset_version=11)

    # Simplify
    import onnx
    import onnxsim
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_file_path)


if __name__ == '__main__':
    export_onnx('./culane_small.onnx')

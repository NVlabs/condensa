import types

import torch
import condensa.tensor as T


class _PatchedOptimizer(object):
    def __init__(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model

    def patch(self):
        __model = self.model

        def __step(self, *args, **kwargs):
            # Apply mask to gradients
            with torch.no_grad():
                for w in __model.modules():
                    if hasattr(w, 'condense'):
                        for pname in w.condense:
                            maskattr = f'__condensa_mask_{pname}'
                            if not hasattr(w, maskattr):
                                raise RuntimeError(
                                    f'Could not retrieve mask for '
                                    f'parameter `{pname}`. Please make sure '
                                    f'to use save_masks() when '
                                    f'applying compression scheme.'
                                )
                            if getattr(w, pname).grad is not None:
                                d_p = getattr(w, pname).grad.data
                                mask = getattr(w, maskattr)
                                if isinstance(mask, list):
                                    for mask_ in mask:
                                        T.apply_mask_inplace(d_p, mask_)
                                else:
                                    T.apply_mask_inplace(d_p, mask)
            # Run original optimizer step
            rval = self.__step(*args, **kwargs)
            # Apply mask to weights
            with torch.no_grad():
                for w in __model.modules():
                    if hasattr(w, 'condense'):
                        for pname in w.condense:
                            maskattr = f'__condensa_mask_{pname}'
                            if not hasattr(w, maskattr):
                                raise RuntimeError(
                                    f'Could not retrieve mask for '
                                    f'parameter `{pname}`. Please make sure '
                                    f'to use save_masks() when '
                                    f'applying compression scheme.'
                                )
                            p = getattr(w, pname).data
                            mask = getattr(w, maskattr)
                            if isinstance(mask, list):
                                for mask_ in mask:
                                    T.apply_mask_inplace(p, mask_)
                            else:
                                T.apply_mask_inplace(p, mask)
            return rval

        self.optimizer.__step = self.optimizer.step
        self.optimizer.step = types.MethodType(__step, self.optimizer)


def patch_optimizer(optimizer, model):
    po = _PatchedOptimizer(optimizer, model)
    po.patch()


def clear_masks(model):
    for m in model.modules():
        if hasattr(m, 'condense'):
            for pname in m.condense:
                maskattr = f'__condensa_mask_{pname}'
                if hasattr(m, maskattr):
                    delattr(m, maskattr)
            delattr(m, 'condense')

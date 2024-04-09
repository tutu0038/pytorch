import pytorch

class Rp(object):
    def __init__(self, raw, replace, **kwargs):
        self.obj = replace
        self.raw = raw

    def __call__(self, *args, **kwargs):
        if not NET_INITTED:
            return self.raw(*args, **kwargs)
        for stack in traceback.walk_stack(None):
            if 'self' in stack[0].f_locals:
                layer = stack[0].f_locals['self']
                if layer in layer_names:
                    log.pytorch_layer_name = layer_names[layer]
                    print('test', layer_names[layer])
                    break
        out = self.obj(self.raw, *args, **kwargs)
        return out

torch.sigmoid = Rp(torch.sigmoid, _sigmoid)

def _sigmoid(raw, input):
    x = raw(input)
    return x
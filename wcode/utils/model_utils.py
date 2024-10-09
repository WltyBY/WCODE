

def get_model_num_paras(model):
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3fM" % (total / 1e6))
    return total
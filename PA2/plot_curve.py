from matplotlib import pyplot as plt

heads = [1, 3, 9, 11, 33]
layers = [1, 2, 4, 6, 8, 10, 12]

plt.figure(figsize=(7, 7))

for head in heads:
    MAE = []
    OBO = []
    out = ''
    for layer in layers:
        with open(f"saved_models/head_{head}_layer_{layer}/test_result.txt", "r") as f:
            content = f.read().split(",")
            mae = float(content[0].split(":")[1])
            obo = float(content[1].split(":")[1])
            MAE.append(mae)
            OBO.append(obo)
            out += "& {:.3f}".format(obo) 

    print(f'{out}')
    # plt.subplot(2, 1, 1)
    # plt.plot(layers, MAE, label=f'num_head={head}')
    # plt.legend()
    # plt.title("MAE")
    # plt.xlim(0, 12)
    # plt.subplot(2, 1, 2)
    # plt.plot(layers, OBO, label=f'num_head={head}')
    # plt.legend()
    # plt.title("OBO")
    # plt.xlim(0, 12)
    plt.plot(layers, OBO, label=f"num_head={head}")
    plt.legend()
    plt.title("OBO")
    plt.xlim(0, 12)
    plt.xlabel("num_layers")
plt.savefig("report/img/OBO_curve.png", bbox_inches='tight', pad_inches=0.0)

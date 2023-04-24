from matplotlib import pyplot as plt

heads = [1, 3, 9, 11]
layers = [1, 2, 4, 6, 8, 10, 12]

plt.figure(figsize=(20, 10))

for head in heads:
    MAE = []
    OBO = []
    for layer in layers:
        with open(f"saved_models/head_{head}_layer_{layer}/test_result.txt", "r") as f:
            content = f.read().split(",")
            mae = float(content[0].split(":")[1])
            obo = float(content[1].split(":")[1])
            MAE.append(mae)
            OBO.append(obo)
    plt.subplot(1, 2, 1)
    plt.plot(layers, MAE, label=f'num_head={head}')
    plt.legend()
    plt.suptitle("MAE")
    plt.subplot(1, 2, 2)
    plt.plot(layers, OBO, label=f'num_head={head}')
    plt.legend()
    plt.suptitle("OBO")
plt.savefig("MAE&OBO_curve.png")

from matplotlib import pyplot as plt

heads = [1, 3, 9, 11]
layers = [1, 2, 4, 6, 8, 10, 12]

fig1 = plt.figure(figsize=(20, 20))
fig2 = plt.figure(figsize=(20, 20))

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
    fig1.plot(layers, MAE, label=f'num_head={head}')
    fig1.legend()
    fig2.plot(layers, OBO, label=f'num_head={head}')
    fig2.legend()
fig1.savefig("MAE_curve.png")
fig2.savefig("OBO_curve.png")

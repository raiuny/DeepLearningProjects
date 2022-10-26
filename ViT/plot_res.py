import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style('ticks', {'font.sans-serif':['simhei', 'Arial']})
sns.set_context('paper')
data = pd.read_csv('ViT/result.csv')
pal = sns.husl_palette(n_colors=4, l = .7)
ls = list(range(1,26))*4
sns.relplot(x = 'epoch', y = 'acc', data = {'epoch': ls, 'acc': data['acc'], 'patch_size': data['patch_size']}, style = 'patch_size', hue ='patch_size', palette = pal, kind='line', markers=True)
plt.savefig('ViT/result.jpg')
plt.show()
# print(data.index+1)


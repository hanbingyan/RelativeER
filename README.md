# Relative Experience Replay

## Files

- `Tabular_Default.py` replicates the results in Calvano et al. (2020, AER). The output implies the profit ratios in Figure 1.

- `Tabular_Relative.py` is tabular Q-learning with relative experience replay in a two-agent setting.

- `Tabular_MultiAgents.py` considers the multi-agent case for `Tabular_Relative.py`.

- `config.py`, `models.py`, `utils.py`, and `DQN.py` are files for deep Q-learning with relative experience replay. They are independent of the three `Tabluar` files.

- `Figs` folder contains jupyter notebooks for figures in the preprint, with outputs generated above. 

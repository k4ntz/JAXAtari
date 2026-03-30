from itertools import groupby
from jaxatari import games
import wandb_workspaces.reports.v2 as wr

ENTITY1 = "openrlbenchmark"
PROJECT1 = "envpool-atari"
ENTITY2 = "raban-emunds-tu-darmstadt"
PROJECT2 = "cleanrl"

report = wr.Report(
    project=PROJECT2,
    title='Comparison of PPO CleanRL: EnvPool vs JAXAtari',
    description="JAXAtari is a JAX-based implementation of Atari games." 
)
env_results = wr.MarkdownBlock("""
EnvPool results are taken from [OpenRLBenchmark](https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-PPO-JAX-EnvPool-s-XLA-part-1---VmlldzoyNTE5Nzcz).
""")

pg = lambda env_id: wr.PanelGrid(
    runsets=[
        wr.Runset(ENTITY1, PROJECT1, "envpool", filters=f"Config('env_id') in ['{env_id}'] and Config('exp_name') in ['ppo_atari_envpool_xla_jax']", groupby=["exp_name"]),
        wr.Runset(ENTITY2, PROJECT2, "jaxatari", filters=f"Config('env_id') in ['{env_id.lower().split('-')[0]}']", groupby=["exp_name"]),
    ],
    panels=[
        wr.LinePlot(x='global_step', y=['charts/avg_episodic_return'], smoothing_factor=0.8, layout=wr.Layout(w=12, h=5)),
        wr.LinePlot(x='_runtime', y=['charts/avg_episodic_return'], smoothing_factor=0.8, layout=wr.Layout(w=12, h=5)),
    ],
)

game_list = ['Pong-v5', "Freeway-v5", "Kangaroo-v5", "Seaquest-v5"]  # Add more games if needed
# report.blocks = report.blocks[:1] + [[wr.H1(f"{g}"), pg(g)] for g in game_list] + report.blocks[1:]
report.blocks = report.blocks[:1] + [env_results] + [item for g in game_list for item in [wr.H1(f"{g}"), pg(g)]] + report.blocks[1:]
# report.blocks = report.blocks[:1] + [wr.H1(f"Freeway-v5"), pg("Freeway-v5")] + [wr.H1(f"Pong-v5"), pg("Pong-v5")] + report.blocks[1:]
report.save()
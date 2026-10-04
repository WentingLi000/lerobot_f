"""Run eval_openloop_new.py for every episode in the configured dataset.

This is a separate entry point: eval_openloop_new.py is not modified.  Keeping
the evaluation implementation in one place also prevents the two scripts from
silently drifting apart.
"""

from pathlib import Path


SOURCE_PATH = Path(__file__).with_name("eval_openloop_new.py")

SELECTION_BLOCK = """if len(all_episodes) <= num_episodes_to_plot:
    target_episodes = all_episodes
else:
    target_episodes = random.sample(all_episodes, num_episodes_to_plot)

print("selected episodes:", target_episodes)"""

ALL_EPISODES_BLOCK = """target_episodes = all_episodes

print("selected all episodes:", target_episodes)"""

OUTPUT_DIR_LINE = 'os.makedirs(plot_dir, exist_ok=True)'
ALL_EPISODES_OUTPUT_DIR = """plot_dir = os.path.join(plot_dir, "all_episodes")
os.makedirs(plot_dir, exist_ok=True)"""


source = SOURCE_PATH.read_text(encoding="utf-8")

if source.count(SELECTION_BLOCK) != 1:
    raise RuntimeError(
        f"Expected exactly one episode-selection block in {SOURCE_PATH}; "
        "the source script may have changed."
    )
if source.count(OUTPUT_DIR_LINE) != 1:
    raise RuntimeError(
        f"Expected exactly one output-directory line in {SOURCE_PATH}; "
        "the source script may have changed."
    )

source = source.replace(SELECTION_BLOCK, ALL_EPISODES_BLOCK)
source = source.replace(OUTPUT_DIR_LINE, ALL_EPISODES_OUTPUT_DIR)

exec(compile(source, str(SOURCE_PATH), "exec"), globals(), globals())

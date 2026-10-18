"""Policies for the runtime: callables from an observation to an action chunk (adapter.labels)."""

from adapter.labels import chunk_actions, load_episode


class ReplayPolicy:
    """A recorded episode played back as a policy: at step t, the chunk labelled at step t of the
    recording, ignoring the observation. Everything around it (anchoring on the live TCP, chunk
    execution, the teleop) is what a trained policy runs through, so a replay that reproduces the
    demonstration tests all of it."""

    def __init__(self, episode, arm, fps, chunk_size):
        self.episode = load_episode(episode, arm, fps)
        self.chunk_size = chunk_size

    @property
    def start(self):
        """The demonstration's first TCP pose (position, quaternion): its actions are relative to the
        TCP, so the replay reproduces it only from where it began."""
        return self.episode.tcp_p[0], self.episode.tcp_q[0]

    def __len__(self):
        return len(self.episode)

    def __call__(self, observation):
        return chunk_actions(self.episode, observation["step"], self.chunk_size)

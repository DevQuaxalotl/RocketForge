from states import GameState
from threading import Lock
import copy


class StateManager:
    def __init__(self):
        self.state = GameState()


class RenderState:
    """Own synthetic ball rotation without mutating published network snapshots."""
    def __init__(self):
        self.published = None
        self.state = None

    def update(self, published, delta_time):
        if published is not self.published:
            previous_ball = self.state.ball_state if self.state is not None else None
            self.state = copy.copy(published)
            self.state.ball_state = copy.copy(published.ball_state)
            ball = self.state.ball_state
            if previous_ball is not None:
                # Network packets can outnumber display frames. Carry synthetic history
                # only until a later packet supplies its own rotation samples.
                if ball.rotation_sequence == previous_ball.rotation_sequence:
                    ball.prev_forward = previous_ball.prev_forward
                    ball.prev_up = previous_ball.prev_up
                    if not ball.has_rot:
                        ball.next_forward = previous_ball.next_forward
                        ball.next_up = previous_ball.next_up
                elif (not ball.rotation_previous_has_rot
                      and ball.rotation_sequence == previous_ball.rotation_sequence + 1):
                    ball.prev_forward = previous_ball.next_forward
                    ball.prev_up = previous_ball.next_up
            self.published = published
        if not self.state.ball_state.has_rot:
            self.state.ball_state.rotate_with_ang_vel(delta_time)
        return self.state


global_state_manager = StateManager()
global_state_mutex = Lock()
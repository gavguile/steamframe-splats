"""OpenXR controller input for free-fly: both thumbsticks, both grips, the right A button.

Bindings are suggested for the Steam Frame controller first, then for the Index
and Touch profiles as fallbacks. Each suggestion stands alone, so a profile the
runtime rejects does not block the others. After the session is focused,
`current_profile()` reports the profile the runtime actually chose.
"""

import logging

import xr

log = logging.getLogger("xr_input")

HANDS = ("/user/hand/left", "/user/hand/right")
PROFILES = (
    "/interaction_profiles/valve/frame_controller",
    "/interaction_profiles/valve/index_controller",
    "/interaction_profiles/oculus/touch_controller",
)


class Controls:
    def __init__(self, instance, session):
        self.instance, self.session = instance, session
        p = lambda s: xr.string_to_path(instance, s)  # noqa: E731
        self.hand = [p(h) for h in HANDS]
        self.action_set = xr.create_action_set(instance, xr.ActionSetCreateInfo(
            action_set_name="flying", localized_action_set_name="Flying", priority=0))

        def action(name, label, kind):
            return xr.create_action(self.action_set, xr.ActionCreateInfo(
                action_type=kind, action_name=name, localized_action_name=label,
                subaction_paths=self.hand))

        self.stick = action("stick", "Move (left) / turn and up-down (right)", xr.ActionType.VECTOR2F_INPUT)
        self.grip = action("boost", "Speed boost", xr.ActionType.FLOAT_INPUT)
        self.reset = action("reset", "Back to start", xr.ActionType.BOOLEAN_INPUT)

        self.accepted = []
        for profile in PROFILES:
            bindings = []
            for h in HANDS:
                bindings += [xr.ActionSuggestedBinding(self.stick, p(f"{h}/input/thumbstick")),
                             xr.ActionSuggestedBinding(self.grip, p(f"{h}/input/squeeze/value"))]
            bindings.append(xr.ActionSuggestedBinding(self.reset, p("/user/hand/right/input/a/click")))
            try:
                xr.suggest_interaction_profile_bindings(instance, xr.InteractionProfileSuggestedBinding(
                    interaction_profile=p(profile), suggested_bindings=bindings))
                self.accepted.append(profile)
            except xr.ResultException as exc:
                log.warning("bindings for %s rejected: %s", profile, exc)
        log.info("bindings accepted for: %s", ", ".join(self.accepted) or "none")
        xr.attach_session_action_sets(session, xr.SessionActionSetsAttachInfo(action_sets=[self.action_set]))

    def poll(self):
        """Returns (left_stick, right_stick, boost, reset) or None if input is not available."""
        try:
            xr.sync_actions(self.session, xr.ActionsSyncInfo(
                active_action_sets=[xr.ActiveActionSet(self.action_set, xr.NULL_PATH)]))
        except xr.ResultException:
            return None  # e.g. session not focused
        sticks, boost = [], False
        for h in self.hand:
            s = xr.get_action_state_vector2f(self.session, xr.ActionStateGetInfo(self.stick, h))
            sticks.append((s.current_state.x, s.current_state.y) if s.is_active else (0.0, 0.0))
            gr = xr.get_action_state_float(self.session, xr.ActionStateGetInfo(self.grip, h))
            boost = boost or (gr.is_active and gr.current_state > 0.5)
        r = xr.get_action_state_boolean(self.session, xr.ActionStateGetInfo(self.reset, self.hand[1]))
        reset = bool(r.is_active and r.current_state and r.changed_since_last_sync)
        return sticks[0], sticks[1], boost, reset

    def current_profile(self):
        out = []
        for name, h in zip(("left", "right"), self.hand):
            try:
                st = xr.get_current_interaction_profile(self.session, h)
                prof = xr.path_to_string(self.instance, st.interaction_profile) if st.interaction_profile else "none"
            except xr.ResultException as exc:
                prof = f"error {exc}"
            out.append(f"{name}: {prof}")
        return "; ".join(out)

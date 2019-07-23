def constant_bp_profile(level=1):
    def bp_profile(f):
        return level
    return bp_profile

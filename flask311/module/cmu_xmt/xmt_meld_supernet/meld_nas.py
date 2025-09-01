import random
import json
import copy

class NASController:
    def __init__(self, prior_path, mutation_rate=0.1, alpha=0.6, weight=0.8):
        """
        Args:
            prior_path: JSON file containing list of subnet dicts with "mask" keys
            mutation_rate: probability of bit flip per path
            alpha: EMA smoothing factor
            weight: trade-off between acc and flops (e.g., 0.8 means 80% acc, 20% flops)
        """
        with open(prior_path, 'r') as f:
            self.subnets = json.load(f)

        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.weight = weight

        self.rewards = {tuple(s["mask"]): 0.0 for s in self.subnets}
        self.trained_masks = set()
        self.flops_cache = {}  # Optional: mask → flops

    def sample_subnets(self, top_k=10):
        """
        Select top-k subnetworks based on reward and apply mutation
        """
        sorted_subnets = sorted(
            self.subnets,
            key=lambda x: self.rewards.get(tuple(x["mask"]), 0),
            reverse=True
        )

        top_subnets = []
        for s in sorted_subnets:
            mask_tuple = tuple(s["mask"])
            if mask_tuple not in self.trained_masks:
                top_subnets.append(s["mask"])
            if len(top_subnets) >= top_k:
                break

        # Mutation
        mutated = []
        for mask in top_subnets:
            new_mask = copy.deepcopy(mask)
            for i in range(len(mask)):
                if random.random() < self.mutation_rate:
                    new_mask[i] = 1 - new_mask[i]  # flip 0↔1
            mutated.append(new_mask)

        return mutated

    def update_rewards(self, subnet_masks, acc_score, flops=None):
        """
        Update EMA reward for each subnet
        """
        for mask in subnet_masks:
            key = tuple(mask)

            # multi-objective reward: weighted acc/flops
            reward = acc_score
            if flops is not None:
                reward = self.weight * acc_score - (1 - self.weight) * flops

            # EMA update
            old_reward = self.rewards.get(key, 0.0)
            new_reward = self.alpha * reward + (1 - self.alpha) * old_reward
            self.rewards[key] = new_reward

            self.trained_masks.add(key)

    def save_rewards(self, path="nas_reward_log.json"):
        """
        Save rewards to JSON file
        """
        log = [{"mask": list(k), "reward": v} for k, v in self.rewards.items()]
        with open(path, "w") as f:
            json.dump(log, f, indent=2)

    def cache_flops(self, mask, flops):
        """
        Optional: Store flops for each mask
        """
        self.flops_cache[tuple(mask)] = flops

    def get_cached_flops(self, mask):
        return self.flops_cache.get(tuple(mask), None)

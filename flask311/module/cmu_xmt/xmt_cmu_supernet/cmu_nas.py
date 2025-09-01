import random
import json
import copy

class NASController:
    def __init__(self, prior_path, mutation_rate=0.1, alpha=0.6, weight=0.8):
        """
        CMU-MOSEI 기반 NAS Controller
        Args:
            prior_path (str): 초기 subnet 후보 리스트 (mask만 있는 JSON)
            mutation_rate (float): 각 path에 대해 flip될 확률
            alpha (float): EMA smoothing factor
            weight (float): reward 계산 시 acc vs flops 비중
        """
        with open(prior_path, 'r') as f:
            self.subnets = json.load(f)  # ex) [{"mask": [0,1,1,1,1,0]}, ...]

        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.weight = weight

        self.rewards = {tuple(s["mask"]): 0.0 for s in self.subnets}
        self.trained_masks = set()
        self.flops_cache = {}

    def sample_subnets(self, top_k=10):
        """
        가장 높은 보상을 받은 top-k subnet을 선택하고 mutation 적용
        Returns:
            list of mutated mask (list of 6 int)
        """
        sorted_subnets = sorted(
            self.subnets,
            key=lambda x: self.rewards.get(tuple(x["mask"]), 0.0),
            reverse=True
        )

        top_subnets = []
        seen = set()

        for s in sorted_subnets:
            key = tuple(s["mask"])
            if key in self.trained_masks or key in seen:
                continue
            top_subnets.append(s["mask"])
            seen.add(key)
            if len(top_subnets) >= top_k:
                break

        # Mutation
        mutated = []
        for mask in top_subnets:
            new_mask = copy.deepcopy(mask)
            for i in range(len(mask)):
                if random.random() < self.mutation_rate:
                    new_mask[i] = 1 - new_mask[i]
            mutated.append(new_mask)

        return mutated

    def update_rewards(self, subnet_masks, acc_score, flops=None):
        """
        선택된 subnet들에 대해 reward 갱신
        Args:
            subnet_masks (List[List[int]]): 학습한 mask 목록
            acc_score (float): 평가 정확도 (acc-7 또는 acc-2)
            flops (float or None): 연산량 (낮을수록 좋음)
        """
        for mask in subnet_masks:
            key = tuple(mask)

            # multi-objective reward 계산
            if flops is not None:
                reward = self.weight * acc_score - (1 - self.weight) * flops
            else:
                reward = acc_score

            # EMA 보상 갱신
            prev = self.rewards.get(key, 0.0)
            updated = self.alpha * reward + (1 - self.alpha) * prev
            self.rewards[key] = updated

            self.trained_masks.add(key)

    def save_rewards(self, path="nas_reward_log.json"):
        """
        보상 로그 저장
        """
        log = [{"mask": list(k), "reward": round(v, 5)} for k, v in self.rewards.items()]
        with open(path, "w") as f:
            json.dump(log, f, indent=2)

    def cache_flops(self, mask, flops):
        """
        mask별 flops 결과 저장
        """
        self.flops_cache[tuple(mask)] = flops

    def get_cached_flops(self, mask):
        return self.flops_cache.get(tuple(mask), None)

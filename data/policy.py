import random
import numpy as np

from data.basic import *


class Policy(object):
    def __init__(self, policy):
        """
        For example, policy is [[(op, mag), (op, mag)]] * Q
        """
        self.policy = policy
        self.queue = []

    def __call__(self, img, mask):
        # maintain queue for CutMix
        self.queue.append((img, mask))
        if len(self.queue) > 10:
            pair = self.queue.pop(0)
        else:
            pair = random.choice(self.queue)

        sub_policy = random.choice(self.policy)
        for op, mag in sub_policy:
            if op == 'CutMix':
                img, mask = apply_cutmix(img, mask, pair[0], pair[1], mag)
            else:
                img, mask = apply_augment(img, mask, op, mag)

        return img, mask
    

class MultiPolicy(object):
    def __init__(self, policies):
        self.policies = []
        for policy in policies:
            self.policies.append(Policy(policy))

    def __call__(self, img):
        imgs = [policy(img) for policy in self.policies] 
        
        return imgs


class DGMultiPolicy(object):
    def __init__(self, policies):
        self.policies = []
        for policy in policies:
            self.policies.append(Policy(policy))

    def __call__(self, sample):
        imgs, labels = [], []
        for policy in self.policies:
            img, label = policy(sample['image'], sample['label'])
            imgs.append(img)
            labels.append(label)
            
        sample['aug_images'] = imgs 
        sample['aug_labels'] = labels
        
        return sample


def parse_policies(policies, config, logger):
    # policies : (M,4(op,mag,op,mag)*5(sub_policy))
    # parsed_policies : [[[(op, mag), (op, mag)]]*5] * M
    exclude_ops = config.CONTROLLER.EXCLUDE_OPS
    L = config.CONTROLLER.L
    NUM_MAGS = config.CONTROLLER.NUM_MAGS
    exclude_op_num = config.CONTROLLER.EXCLUDE_OPS_NUM
    
    al = augment_list()
    if len(exclude_ops) > 0:
        al = [op for op in al if op[0].__name__ not in exclude_ops]
        logger.info(exclude_ops)
    elif exclude_op_num > 0:
        for i in range(exclude_op_num):
            seed = np.random.randint(0, 65536) * config.SEED
            random.seed(seed)
            random.shuffle(al)
            exclude_ops = al.pop(0)
            config.CONTROLLER.EXCLUDE_OPS.append(exclude_ops[0].__name__)
            logger.info(exclude_ops[0].__name__)
    
    M, S = policies.shape
    S = S // (L * 2)
    parsed_policies = []
    for i in range(M):
        parsed_policy = []
        for j in range(S):
            policy = []
            for k in range(L):
                policy.append((al[policies[i][2*L*j+k*2]][0].__name__, policies[i][2*L*j+k*2+1]/(NUM_MAGS-1)))
            parsed_policy.append(policy)
        parsed_policies.append(parsed_policy)
    
    return parsed_policies

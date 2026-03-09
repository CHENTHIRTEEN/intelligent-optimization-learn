import numpy as np
np.random.seed(2026)
import matplotlib.pyplot as plt

best_value = []
class Particle:
    def __init__(self, dim : int, bounds : list[float, float]):
        """_summary_

        Args:
            dim (int): 搜索空间（粒子）的维度
            bounds (List[float, float]): 搜索空间的上下界
        """
        self.pos = np.random.uniform(bounds[0], bounds[1], dim) # 粒子的初始位置
        self.v = np.random.uniform(-1, 1, dim) # 粒子的初始速度（分方向）
        self.best_pos = self.pos.copy() # 最佳位置
        self.best_value = float('inf') # 最优解
    
    def update_v(self, global_best_pos: np.array, 
                 inertia_weight: float = 0.5, 
                 cognitive_weight: float = 1.0,
                 social_weight: float = 1.0):
        """_summary_

        Args:
            global_best_pos (np.array): 当前最优粒子位置
            inertia_weight (float, optional): 惯性系数w. Defaults to 0.5.
            cognitive_weight (float, optional): 学习系数C_1. Defaults to 1.0.
            social_weight (float, optional): 学习系数C_2. Defaults to 1.0.
        """
        r_1 = np.random.rand(len(self.pos))
        r_2 = np.random.rand(len(self.pos))
        cognitive_component = cognitive_weight * r_1 * (self.best_pos - self.pos)
        social_component = social_weight * r_2 * (global_best_pos - self.pos)
        self.v = inertia_weight * self.v + cognitive_component + social_component
        
    def update_pos(self, bounds : list[float, float]):
        self.pos += self.v
        self.pos = np.clip(self.pos, bounds[0], bounds[1])
        
class PSO:
    def __init__(self, nums : int, dim : int, bounds : list[float, float]):
        """初始化算法

        Args:
            nums (int): 粒子个数
            dim (int): 搜索空间维度
            bounds (List[float, float]): 搜索空间边界
        """
        self.particles = [Particle(dim, bounds) for _ in range(nums)]
        self.global_best_position = np.random.uniform(bounds[0], bounds[1], dim)
        self.global_best_value = float('inf')
        
    def optim(self, obj_fun, max_iter : int = 100):
        for i in range(max_iter):
            for particle in self.particles:
                value = obj_fun(particle.pos)
                if value < particle.best_value:
                    particle.best_value = value
                    particle.best_pos = particle.pos.copy()
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_pos = particle.pos.copy()
            best_value.append(self.global_best_value)

            for particle in self.particles:
                particle.update_v(self.global_best_pos)
                particle.update_pos(bounds)
                
if __name__ == "__main__":
    def fun(x):
        return np.sum(np.cos(x) + x ** 2)
    
    num_particles = 20
    dim = 2
    bounds = [-5, 5]
    pso = PSO(num_particles, dim, bounds)
    pso.optim(fun)
    print("Global Best Position:", pso.global_best_position)
    print("Global Best Value:", pso.global_best_value)
    plt.plot(best_value)    
    plt.xlabel('Iteration')
    plt.ylabel('Global Best Value')
    plt.title('PSO Optimization Progress')
    plt.show()
# Resource: https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/

import random
import math

def func1(x):
    total=0
    for i in range(len(x)):
        total += abs(x[i]**2 - 12*x[i] + 12)
    return total

class Particle:
    def __init__(self, init_pos):
        self.position = init_pos
        self.velocity = []
        self.pbest_pos = []
        self.pbest_error = -1
        self.curr_error = -1
        self.num_dimensions = len(init_pos)

        for i in range(0, self.num_dimensions):
            self.velocity.append(random.uniform(-1, 1))

    def calculate_cost(self, costFunction):
        self.curr_error = costFunction(self.position)

        if(self.curr_error < self.pbest_error or self.pbest_error == -1):
            self.pbest_pos = self.position
            self.pbest_error = self.curr_error

    def updateVelocity(self, inertia, cog_const, soc_const, gbest_pos):
        for i in range(0, self.num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive = cog_const * r1 * (self.pbest_pos[i] - self.position[i])
            vel_social= soc_const * r2 * (gbest_pos[i] - self.position[i])
            self.velocity[i] = inertia * self.velocity[i] + vel_cognitive + vel_social

    def updatePosition(self, bounds):
        for i in range(0, self.num_dimensions):
            self.position[i] = self.position[i] + self.velocity[i]

            if self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]

class PSO:
    def __init__(self, costFunction, init_pos, bounds, num_particles, epoch):
        num_dimensions = len(init_pos)
        gbest_pos = []
        gbest_error = -1
        inertia = 0.5
        cog_const = 1
        soc_const = 2

        swarm = []

        for i in range(0, num_particles):
            swarm.append(Particle(init_pos))

        print("")
        for timestep in range(epoch):
            print("Timestep: %d" % timestep)
            for i in range(0, num_particles):
                print(swarm[i].position)
                swarm[i].calculate_cost(costFunction)

                if swarm[i].curr_error < gbest_error or gbest_error == -1:
                    gbest_pos = list(swarm[i].position)
                    gbest_error = float(swarm[i].curr_error)

                swarm[i].updateVelocity(inertia, cog_const, soc_const, gbest_pos)
                swarm[i].updatePosition(bounds)
            #print("Gbest Position: " + str(gbest_pos))
            #print("Gbest Error: " + str(gbest_error))

        # Revisit Later
        print("---------------------------------")
        print("Final:")
        print("Gbest Position: " + str(gbest_pos))
        print("Gbest Error: " + str(gbest_error))

# Check Later
if __name__ == "__PSO__":
    main()

initial = [5,5]
bounds = [(-10, 10), (-10, 10)]
PSO(func1, initial, bounds, num_particles = 50, epoch = 100)

import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.reset()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    # adjoining = nearby
    # update the state; quite straight-forward to implement 
    def discretize(self, state):
        # GIVEN: state= [snake_head_x, snake_head_y, snake_body, food_x, food_y] 

        #i nitialization with default sets
        tmp_snake_body = []
        
        # discrete state into its element in the order they are saved within the state\
        # modification: I removed floor functions and use constants instead of using magic number constants like utils.GRID_SIZE 
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state
        snake_head_x = snake_head_x/40
        snake_head_y = snake_head_y/40
        for i, j in snake_body:
            tmp_snake_body.append(((i/40), (j/40)))
        food_x = food_x/40
        food_y = food_y/40

        # update adjoining_wall
        # sanity check; out of boundary
        # modification: also simpliekd lists into single variables so it doesnt have to be accessed with index which causes the time much longer
        adjoining_wall_x = 0
        adjoining_wall_y = 0 
        if (snake_head_x < 1) or (snake_head_x > 13) or (snake_head_y < 1) or (snake_head_y > 13):
            adjoining_wall_x = 0
            adjoining_wall_y = 0
        else:    
            if snake_head_x == 1:
                adjoining_wall_x = 1
            elif snake_head_x == 12:
                adjoining_wall_x = 2
            
            if snake_head_y == 1:
                adjoining_wall_y = 1
            elif snake_head_y == 12:
                adjoining_wall_y = 2    

        # update food_dir
        food_dir_x = 0 
        food_dir_y = 0
        if (food_x - snake_head_x) < 0:
            food_dir_x = 1
        elif (food_x - snake_head_x) > 0:
            food_dir_x = 2  
        if (food_y - snake_head_y) < 0:
            food_dir_y = 1
        elif (food_y - snake_head_y) > 0:
            food_dir_y = 2
        
        # update snake_body
        adjoining_body_top = 0
        adjoining_body_bottom = 0 
        adjoining_body_left = 0
        adjoining_body_right = 0
        #body_top
        if ((snake_head_x, snake_head_y-1) in tmp_snake_body):
            adjoining_body_top = 1              
        #body_bottom 
        if ((snake_head_x, snake_head_y+1) in tmp_snake_body):
            adjoining_body_bottom = 1           
        #body_left
        if ((snake_head_x-1, snake_head_y) in tmp_snake_body):
            adjoining_body_left = 1        
        #body_right
        if ((snake_head_x+1, snake_head_y) in tmp_snake_body):
            adjoining_body_right = 1        

        return (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)



    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''

        # 1. discretize the state into its elements 
        ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8 = self.discretize(state)
    
        # 2. update Q-Table, but only when train is TRUE and self.s and self.a are not NONE(no previous states, so very initial state)
        if (self._train) and (self.s is not None) and (self.a is not None):
            #another discretization, but with old state 
            s1, s2, s3, s4, s5, s6, s7, s8 = self.s
            
            # find max_Q using current state, which is self.s and current action, which is action 
            max_Q = -np.inf

            #get q values of new state and all possible new actions 
            for action in self.actions:
                if (self.Q[ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,action] > max_Q):
                    max_Q = self.Q[ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,action]
            
            # calculate the alpha; alpha = C/(C+N(s,a))
            alpha = self.C / (self.C + self.N[s1,s2,s3,s4,s5,s6,s7,s8,self.a])
            
            # calculate the reward 
            reward = -0.1
            if dead :
                reward = -1
            if (points - self.points) > 0:
                reward = 1

            # finally update Q; Q(s,a) = Q(s,a)+α(R(s)+gamma*max_a′Q(s′,a′)−Q(s,a))    
            self.Q[s1,s2,s3,s4,s5,s6,s7,s8,self.a] += alpha * (reward + self.gamma * max_Q - self.Q[s1,s2,s3,s4,s5,s6,s7,s8,self.a])
        
        # 3. sanity check: dead case 
        if dead:
            self.reset()
            return 
        # if not dead case, save new points and new state accordingly    
        else:
            self.s = (ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8)
            self.points = points
        
        # 4. find the best action; this is where we explore, and compute f(q,n) for all actions
        # modification: removed extra method implemented for exploration, but just merged it within the act function instead 
        best_action = 0
        pos_one = 1
        argmax_f = -np.inf

        # exploration policy: f(q,n) = 1 if n < self.Ne else q 
        for action in self.actions:
            n = self.N[ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,action]

            # main f function: f(q,n)
            if n < self.Ne :
                if argmax_f <= pos_one:
                    argmax_f = pos_one
                    best_action = action
            else:
                q = self.Q[ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,action]
                if argmax_f <= q:
                    argmax_f = q
                    best_action = action
        
        # 5. update action by replacing with best_action; save new action
        self.a = best_action

        # 6. update N table incrementing by one  
        self.N[ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,best_action] += 1

        # 7. return our best action or new action; if this works, my last assignment of the university is done finally :)
        return best_action


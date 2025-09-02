import cv2
import numpy as np
import os

from math import sqrt

class Plotter:
    def __init__(self, envParams):
        
        self.set_params(envParams)

        self.rows = self.gridParams['rows']
        self.cols = self.gridParams['cols']
        self.step = 2*(int(sqrt(self.gridParams['cell_size'])) // 2) + 1
        self.height = self.rows * self.step
        self.width = self.cols * self.step
        self.indent = self.step // 2

        self.image = np.ones((self.height + 2*self.indent, self.width + 2*self.indent, 3), dtype=np.uint8) * 255  # Create a white screen
        self.add_grid(step=self.step)

        self.ids = self.renderParams['plot_agent_ids']
        self.manual_progression = self.renderParams['manual_plot_progression']

    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def clear_image(self):
        self.image = np.ones((self.height + 2*self.indent, self.width + 2*self.indent, 3), dtype=np.uint8) * 255
        self.add_grid(step=self.step)

    def plot_agent(self, agent, level, color=(255, 0, 0), radius=4):
        id = agent.id
        center = self.step * agent.gridPosition_ + np.array([(self.step // 2) + 1, (self.step // 2) + 1]) + np.array([self.indent, self.indent])
        if radius + level > 5:
            radius = 5
        else:
            radius = radius + level

        if level == 1:
            cv2.circle(self.image, (int(center[0]), int(center[1])), radius, (0,255,0), -1)
        elif level == 2:
            cv2.circle(self.image, (int(center[0]), int(center[1])), radius, (0,0,255), -1)
        elif level == 3:
            cv2.circle(self.image, (int(center[0]), int(center[1])), radius, (0,165,255), -1)
        elif level == 4:
            cv2.circle(self.image, (int(center[0]), int(center[1])), radius, (128,0,128), -1)
        elif level == 5:
            cv2.circle(self.image, (int(center[0]), int(center[1])), radius, (255,204,51), -1)
        else:
            cv2.circle(self.image, (int(center[0]), int(center[1])), radius, color, -1)

        # adding ID
        # Add X and Y coordinates as text
        # text = f'({id})'
        # text_position = (int(center[0]) + 10, int(center[1]) + 10)  # Adjust the position for text
        # if self.ids:
        #     self.add_text(text, text_position, color, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1)

    def plot_target(self, target, radius=7):
        center = self.step * target.gridLocation_ + np.array([(self.step // 2) + 1, (self.step // 2) + 1]) + np.array([self.indent, self.indent])
        # size = radius + 2
        # if target.movement is not None:
        #     if target.direction == 0: # Left
        #         direction = np.pi
        #     elif target.direction == 1: # Right
        #         direction = 0
        #     elif target.direction == 2: # Up
        #         direction = np.pi / 2
        #     elif target.direction == 3: # Down
        #         direction = 3 * np.pi / 2
        
        #     # Calculate the vertices of the triangle based on the direction
        #     angle1 = direction - np.pi / 4
        #     angle2 = direction + np.pi / 4

        #     vertex1 = (int(center[0] + (size+2) * np.cos(direction)),
        #                int(center[1] + (size+2) * np.sin(direction)))
        vertex1 = center + np.array([-1*radius // 2, radius // 2])
        #     vertex2 = (int(center[0] + size * np.cos(angle1)),
        #                int(center[1] + size * np.sin(angle1)))
        vertex2 = center + np.array([radius // 2, radius // 2])
        #     vertex3 = (int(center[0] + size * np.cos(angle2)),
        #                int(center[1] + size * np.sin(angle2)))
        vertex3 = center + np.array([0 , -1*radius // 2])
        #     # Draw the filled triangle
        triangle_pts = np.array([vertex1, vertex2, vertex3], np.int32)
        cv2.fillPoly(self.image, [triangle_pts], color=(0, 0, 255))

        # else: # targets not mobile
        #cv2.circle(self.image, (int(center[0]), int(center[1])), radius, (0,0,255), 1)
    def plot_cleared_target(self, target, radius=7):
        center = self.step * target.gridLocation_ + np.array([(self.step // 2) + 1, (self.step // 2) + 1]) + np.array(
            [self.indent, self.indent])
        vertex1 = center + np.array([-1 * radius // 2, radius // 2])
        vertex2 = center + np.array([radius // 2, radius // 2])
        vertex3 = center + np.array([0, -1 * radius // 2])
        triangle_pts = np.array([vertex1, vertex2, vertex3], np.int32)
        cv2.polylines(self.image, [triangle_pts], isClosed=True, color=(0, 0, 255), thickness=1)

    def add_lines(self, token_holder, member, level):
        th_center = self.step * token_holder.gridPosition_ + np.array([(self.step // 2) + 1, (self.step // 2) + 1]) + np.array([self.indent, self.indent])
        #th_center = sqrt(self.gridParams['cell_size']) * (token_holder.gridPosition_ + np.array([0.5,0.5]))
        m_center = self.step * member.gridPosition_ + np.array([(self.step // 2) + 1, (self.step // 2) + 1]) + np.array([self.indent, self.indent])
        #m_center = sqrt(self.gridParams['cell_size']) * (member.gridPosition_ + np.array([0.5,0.5]))

        if level == 1: color = (255, 0, 0)
        elif level == 2: color = (0, 255, 0)
        elif level == 3: color = (0, 0, 255)
        elif level == 4: color = (0, 165, 255)
        elif level == 5: color = (128, 0, 128)
        
        cv2.line(self.image,(int(th_center[0]), int(th_center[1])),(int(m_center[0]), int(m_center[1])),color)

    def add_text(self, text, position=(10, 30), color=(0, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=1): 
        # Reseting text region
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        image_height, _ = self.image.shape[:2]
        clear_region = ((position[0], position[1] - text_height), (position[0] + text_width, position[1]))
        cv2.rectangle(self.image, clear_region[0], clear_region[1], (255, 255, 255), -1)

        # Displaying text
        cv2.putText(self.image, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    def add_grid(self, step, color=(160, 160, 160), thickness=1):
        rows = self.rows
        cols = self.cols
        for i in range(0, rows + 1):
            y = i * step + self.indent
            cv2.line(self.image, (self.indent, y), (self.width + self.indent, y), color, thickness)

        for i in range(0, rows):
            y = i * step + self.indent + (step // 2) + 1
            cv2.line(self.image, (self.indent, y), (self.width + self.indent, y), (224, 224, 224), thickness)

        # Draw vertical grid lines
        for i in range(0, cols + 1):
            x = i * step + self.indent
            cv2.line(self.image, (x, self.indent), (x, self.height + self.indent), color, thickness)

        for i in range(0, cols):
            x = i * step + self.indent + (step // 2) + 1
            cv2.line(self.image, (x, self.indent), (x, self.height + self.indent), (224, 224, 224), thickness)

    def display(self, window_name='Point Plotter', text=''):
        self.add_text(text)
        cv2.imshow(window_name, self.image)
        if self.manual_progression:
            cv2.waitKey(0) # (0) if you want user input
        else:
            cv2.waitKey(500)
    
    def end(self):
        cv2.destroyAllWindows()

    def save(self, filename, path):
        full_path = os.path.join(path, filename).replace('\\','/')
        cv2.imwrite(full_path, self.image)
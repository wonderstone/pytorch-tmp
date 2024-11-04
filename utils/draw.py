from time import sleep
import matplotlib.pyplot as plt

class DynamicPlot:
    def __init__(self, title="Dynamic Multi-Line Plot", x_label="x", y_label="y"):
        self.data = {}  # Dictionary to store x and y data for multiple lines

        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)
        self.ax.legend()
        self.ax.grid(True)

    def add_line(self, line_id, color="blue"):
        """Add a new line to the plot"""
        if line_id not in self.data:
            self.data[line_id] = {'x': [], 'y': [], 'line': self.ax.plot([], [], label=f"Line {line_id}", color=color)[0]}
            self.ax.legend()

    def add_data(self, line_id, x, y):
        """Add new x and y data to a specific line"""
        if line_id in self.data:
            self.data[line_id]['x'].append(x)
            self.data[line_id]['y'].append(y)
            self.update_plot()  # Update the plot after adding new data

    def update_plot(self,interval=0.01):
        """Update the plot to display new data"""
        for line_id, line_data in self.data.items():
            # Update line data
            line_data['line'].set_data(line_data['x'], line_data['y'])
        
        # Adjust the axis limits based on the data
        all_x = [x for line_data in self.data.values() for x in line_data['x']]
        all_y = [y for line_data in self.data.values() for y in line_data['y']]
        if all_x and all_y:
            self.ax.set_xlim(min(all_x), max(all_x))
            self.ax.set_ylim(min(all_y), max(all_y))
        
        # Redraw the plot
        plt.draw()
        plt.pause(interval)
    
    def show(self):
        plt.show()

if __name__ == "__main__":
    # Create a DynamicPlot instance
    plot = DynamicPlot()

    # Add multiple lines
    plot.add_line('line1', color="blue")
    plot.add_line('line2', color="red")

    # Simulate data and update the plot
    for i in range(1, 21):  # Add 20 sets of data points
        x = i
        y1 = 2 * x - 5  # Example: generate y = 2 * x - 5 for line1
        y2 = -2 * x + 5  # Example: generate y = -2 * x + 5 for line2
        plot.add_data('line1', x, y1)  # Add data to line1
        plot.add_data('line2', x, y2)  # Add data to line2
        sleep(0.1)  # Simulate delay between data points

    plot.show()  # Display the plot



import os
import csv
from src.shared.misc_functions import conv_quat_to_euler
import matplotlib.pyplot as plt

# two functions for computing/plotting rat head direction data
def quat_file_to_euler(filename='head_orientation.csv', nf='True'):  ##returns newfilename
    new_filename = filename.replace('.csv', 'in_euler_angles.csv')
    if os.path.exists(filename):
        print('File exists')
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            if nf == 'True':  # do you want to create a new file
                with open(new_filename, 'w', newline='') as nf:
                    writer = csv.writer(nf)
                    header = []
                    header.append('Time Stamp (ms)')
                    header.append('x')
                    header.append('y')
                    header.append('z')
                    writer.writerow(header)
                    next(f)
                    for line in reader:
                        euler_angles = conv_quat_to_euler(line)
                        writer.writerow(euler_angles)
                return new_filename
            else:
                matrix = []
                next(f)
                for line in reader:
                    euler_angles = conv_quat_to_euler(line)
                    matrix.append(euler_angles)
                return matrix


def graph_movement(filename='head_orientation_in_euler_angles.csv', plot_name='movement_plot.png'):  ##eulerAngle file
    if 'in_euler_angles.csv' not in filename and '.csv' in filename:
        filename = quat_file_to_euler(filename)
    elif '.csv' not in filename:
        print('!!! ERROR: Invalid file')
        return

    if os.path.exists(filename):
        print('File exists')
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            y = []
            avg_angle = []
            time = []
            next(f)  ##skip header line
            for line in reader:
                if len(line) != 4:
                    print('!!! ERROR: Invalid file') #FIXME
                    return
                euler_anglesum = (float(line[1]) + float(line[2]) + float(
                    line[3])) / 3  # FIXME change to difference between angles instead of averaging the angles
                avg_angle.append(euler_anglesum)
                time.append(float(line[0]))
            count = 1  ##skips first line
            while count < len(avg_angle):
                delta_angle = abs((avg_angle[count]) - avg_angle[count - 1])
                delta_time = abs(time[count] - avg_angle[count - 1])
                y.append(delta_angle / delta_time)
                count += 1
            '''
            FIXME
            make an array and take the diff between the rows so you have three columns
            figure out how you want to represent them as one value and graph
            '''
            x = list(time[1:])  ##skips first time
            y = list(y)
            plt.plot(x, y)
            plt.xlabel('time(ms)')
            plt.ylabel('angle change over time (rad/s)')
            plt.title('movement over time')
            plt.show()
            plt.savefig(plot_name)
    else:
        print('!!! ERROR: File not found') #FIXME
        return
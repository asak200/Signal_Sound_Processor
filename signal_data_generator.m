clear;
clc;

file_path = 'the/path/to/where/the/file/is/saved'


% Define your variables
sampling_freq = 50;
duration = 10;

format long

t = 0:1/sampling_freq:duration;

% here is the signal to be generated
data_set = 3*cos(2*pi.*t)...
    + 1*cos(2*pi*3.*t+pi/2)...
    + 6*cos(2*pi*6.*t+ 2)...
    + 2*cos(2*pi*5.*t+ 2);

% Open a file for writing

fid = fopen(file_path, 'w');

% Write variable1 (1D vector)
fprintf(fid, 'data_set:\n');
fprintf(fid, '%f ', data_set);
fprintf(fid, '\n');

% Write variable2 (2D matrix)
fprintf(fid, 'sampling_freq:\n');
fprintf(fid, '%f\n', sampling_freq);

% Write variable3 (scalar)
fprintf(fid, 'duration:\n');
fprintf(fid, '%f\n', duration);

% Close the file
fclose(fid);

plot(t, data_set)


#!/usr/bin/env python2
"""
kafe2 example: Linear Regression
================================

The simplest, and also the most common use case of a fitting framework
lies in linear regression, also known as a line fit:
A linear function of the form f(x) = a * x + b is made to align with
a series of xy data points that have some uncertainty along the y-axis.
"""

from kafe2 import XYContainer, XYFit, XYPlot
import matplotlib.pyplot as plt

# Create an XYContainer object to hold the xy data for the fit
xy_data = XYContainer(x_data=[1.0, 2.0, 3.0, 4.0],
                      y_data=[2.3, 4.2, 7.5, 9.4])
# x_data and y_data are combined depending on their order.
# The above translates to the points (1.0, 2.3), (2.0, 4.2), and (4.0, 9.4)

# Important: Specify y-uncertainties for the data
xy_data.add_simple_error(axis='y', err_val=0.4)

# Create an XYFit object from the xy data container
# By default, a linear function f=a*x+b will be used as the model function.
line_fit = XYFit(xy_data=xy_data)

# Perform the fit: Find values for a and b that minimize the
#     difference between the model function and the data.
line_fit.do_fit() #This will throw an exception if no errors were specified

# Optional: Print out a report on the fit results on the console
line_fit.report()

# Optional: Create a plot of the fit results using XYPlot
plot = XYPlot(fit_objects=line_fit) # Create a kafe2 plot object
plot.plot() # do the plot
plot.show_fit_info_box() # Optional: Add numerical fit results to the image

# show the fit result
plt.show()



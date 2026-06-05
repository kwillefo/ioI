#>>> A. IMPORT REQUIRED MODULES

import cv2
from google.colab.patches import cv2_imshow
import ipywidgets as widgets
from ipywidgets import interact, IntSlider, FloatSlider
import math
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import numpy
from numpy import random as npr
rc('animation', html='jshtml')
import pandas as pd
import string
import warnings
warnings.filterwarnings('ignore')

#>>> B. DEFINE GLOBAL VARIABLES

#> AXIS LIMITS
xLim = [-1, 1]
yLim = [-0.5, 0.5]

#> COLORS
numIndexColors = 1001
halfIndexColors = round(numIndexColors / 2)
cMapIndex = plt.colormaps['GnBu'].resampled(numIndexColors)

numThemeColors = 8
cMapTheme = plt.colormaps['gist_earth'].resampled(numThemeColors)

#> TEXT
tbox = dict(facecolor = cMapTheme(6), edgecolor = cMapTheme(0),
            boxstyle = 'round, pad = 0.3', linewidth = 1, alpha = 1.0)
plt.rcParams.update({'font.size': 10}) # global font size
letters = list(string.ascii_uppercase)
textY = yLim[0] * 0.85

#>>> C. FUNCTION DEFINITIONS

#> COMPUTATIONS

def computeDeviationThick(n1, theta1, nPrism, A):

  # refraction @ first interface
  n1p = nPrism # emergent refractive index
  theta1p = round(numpy.degrees(math.asin((n1 * math.sin(numpy.radians(theta1)))
          / n1p)), 3)

  # finding second incident angle
  # the sum of the internal angles is equivalent to the apical angle
  theta2 = round(A - theta1p, 3)

  # refraction @ second interface
  n2 = nPrism # incident refractive index
  n2p = n1 # emergent refractive index
  try:
    theta2p = round(numpy.degrees(math.asin((n2 * math.sin(numpy.radians(theta2)))
            / n2p)), 3)
    devthick = round(theta1 + theta2p - A, 3)

  except:
    thetaCrit = round(numpy.degrees(math.asin(n2p/n2)))
    devthick = numpy.nan

  return devthick

def computeDisplacement(n, theta1Deg, np, t):
  theta1Rad = math.radians(theta1Deg) # convert incident angle to radians
  sinAngle1 = (n * math.sin(theta1Rad)) / np # first step of computing thetaP1
  if abs(sinAngle1) <= 1: # second step; take arcsin of sinAngle1
    thetaP1Rad = math.asin(sinAngle1)

  d = t * math.sin(theta1Rad - thetaP1Rad) / math.cos(thetaP1Rad)
  return d

def computeLensForm(F1, F2):

  powerList = [F1, F2]
  powerSum = sum(powerList)
  if any(F == 0 for F in powerList):
    if powerSum < 0:
      print('The lens is plano-concave.')
    elif powerSum > 0:
      print('The lens is plano-convex.')
    elif powerSum == 0:
      print('This is a parallel-sided block.')

  if F1 * F2 > 0:
    if powerSum < 0:
      print('The lens is biconcave.')
    elif powerSum > 0:
      print('The lens is biconvex.')
  elif F1 * F2 < 0:
    if powerSum < 0:
      print('The lens is a negative meniscus.')
    if powerSum > 0:
      print('The lens is a positive meniscus.')

def computeThickPrismRefractions(n1, theta1, A, nPrism): 

  print('The first incident angle is:', theta1, '°.')
  n1p = nPrism # emergent refractive index
  theta1p = round(numpy.degrees(math.asin((n1 * math.sin(numpy.radians(theta1)))
           / n1p)), 3)
  print('The first emergent angle is:', theta1p, '°.')

  # finding second incident angle
  # the sum of the internal angles is equivalent to the apical angle
  theta2 = round(A - theta1p, 3)
  print('The second incident angle is:', theta2, '°.')

  # refraction @ second interface
  # light bends away from this normal
  n2 = nPrism # incident refractive index
  # theta2 found above
  n2p = n1 # emergent refractive index
  try:
    theta2p = round(numpy.degrees(math.asin((n2 *math.sin(numpy.radians(theta2)))
            / n2p)), 3)
    print('The second emergent angle is:', theta2p, '°.')

    # deviation of a thick prism
    # found by comparing sum of external angles to the apical angle
    devthick = round(theta1 + theta2p - A, 3)
    print('The resultant deviation is:', devthick, '°.')

  except:
    thetaCrit = round(numpy.degrees(math.asin(n2p/n2)))
    print('The critical angle of this prism is:', thetaCrit, '°.')
    print('That is why Total Internal Reflection occurred at the second interface.')

def computeVergence(n, r):
  V = round(n/r, 3)
  return V

def computeWaveStats(v, lam, f):
  if v == None:
    v = lam * f
    answer = v
  elif lam == None:
    lam = v/f
    answer = lam
  elif f == None:
    f = v / lam
    answer = f
  return round(answer, 3)

#> DRAWINGS

def drawCurvedInterface(r, n, np):

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_aspect('equal', adjustable='box')

    rMag = abs(r)
    h = rMag * 0.8
    yVals = numpy.linspace(-h, h, 100)

    if r > 0:
        xVals = r - numpy.sqrt(r**2 - yVals**2)
        C = (r, 0)
    else:
        xVals = r + numpy.sqrt(r**2 - yVals**2)
        C = (r, 0)

    F = round(safeDivide(np - n, r), 3) # power
    print('The power of this surface is: ', F, 'D.')
    if F > 0:
        print('The surface is convex.')
    elif F < 0:
        print('The surface is concave.')

    incColor = getInterpolatedColor(n)
    emColor = getInterpolatedColor(np)

    # dynamic adjustment of plot limits
    xLim = max(abs(xVals).max(), rMag, abs(C[0])) * 1.5
    yLim = h * 1.5
    ax.set_xlim(-xLim, xLim)
    ax.set_ylim(-yLim, yLim)
    minX, maxX = ax.get_xlim()
    minY, maxY = ax.get_ylim()

    # incident medium
    incPoints = [(xVals[-1], h), (minX, h), (minX, -h), (xVals[0], -h)]
    incPoints.extend(zip(xVals, yVals))
    incPatch = patches.Polygon(incPoints, closed=True, facecolor=incColor, alpha=0.3, lw=0)
    ax.add_patch(incPatch)
    ax.text(-rMag * 1.2, h * 0.9, f'$n = {n}$', ha = 'center', bbox = tbox)

    # emergent medium
    emPoints = [(xVals[-1], h), (maxX, h), (maxX, -h), (xVals[0], -h)]
    emPoints.extend(zip(xVals, yVals))
    emPatch = patches.Polygon(emPoints, closed=True, facecolor=emColor, alpha=0.3, lw=0)
    ax.add_patch(emPatch)
    ax.text(rMag * 1.2, h * 0.9, f"$n' = {np}$", ha='center', bbox = tbox)

    # curved surface
    ax.plot(xVals, yVals, color = cMapTheme(1), lw = 4)

    # plot light direction convention
    lightDirY = minY + (maxY - minY) * 0.05
    ax.quiver(minX, lightDirY, maxX - minX, 0, color = cMapTheme(0), 
             scale_units = 'xy', scale = 1, width = 0.003)

    # plot optical axis
    ax.plot([minX, maxX], [0, 0], color = cMapTheme(0), linewidth = 0.5, zorder = 0)

    #> RAYS FOR EACH VERTICAL POSITION ON CURVED SURFACE
    yPosList = [h * 0.5, h * -0.5]
    uEmList = list()
    for yPos in yPosList:
        try:
            if r > 0:
                xPos = r - numpy.sqrt(r**2 - yPos**2)
            else: # r < 0
                xPos = r + numpy.sqrt(r**2 - yPos**2)
        except ValueError: # Added to complete the try block and handle potential sqrt of negative number
            continue # Skip this iteration if yPos is out of range for r

        # incident ray
        ax.plot([minX, xPos], [yPos, yPos], color= cMapTheme(0),
                linestyle = '-', lw = 2)

        # normal
        xToC = xPos - C[0]
        yToC = yPos - C[1]
        rLen = numpy.sqrt(xToC**2 + yToC**2)

        if rLen > 0:
            unitVecX = xToC / rLen
            unitVecY = yToC / rLen

            # Normal segment starts before the interface and ends after it
            normSegLength = rMag * 0.1

            nx0 = xPos - unitVecX * normSegLength
            ny0 = yPos - unitVecY * normSegLength
            nxF = xPos + unitVecX * normSegLength
            nyF = yPos + unitVecY * normSegLength

            ax.plot([nx0, nxF], [ny0, nyF], color= cMapTheme(0), linestyle=':', lw=1)

            if r > 0:
                # for r > 0, normal points from (xPos, yPos) to center
                localNormX = C[0] - xPos
                localNormY = C[1] - yPos
            else:
                # for r < 0, C normal points from center to (xPos, yPos)
                localNormX = xPos - C[0]
                localNormY = yPos - C[1]

            # Normalize this normal vector
            localNormLen = numpy.sqrt(localNormX**2 + localNormY**2)
            if localNormLen == 0: continue
            localNormX /= localNormLen
            localNormY /= localNormLen

            # angle of normal from optical axis
            phiNormal = math.atan2(localNormY, localNormX)
            # angle of incident ray from optical axis
            uInc = 0.0

            # calculate angle of incident ray relative to normal
            theta = uInc - phiNormal
            # ensure it's within (-pi, pi]
            theta = math.atan2(math.sin(theta), math.cos(theta))

            # Snell's Law
            leftSideofEq = (n * math.sin(theta)) / np

            thetap = math.asin(leftSideofEq)

            # angle of emergent ray (relative to optical axis) is phiNormal + thetap
            uEm = phiNormal + thetap
            uEmList.append(uEm)

            # Extend the ray outwards. Assume a reasonable length for visualization.
            emLength = maxX - xPos
            emxF = xPos + emLength * math.cos(uEm)
            emyF = yPos + emLength * math.sin(uEm)

            ax.plot([xPos, emxF], [yPos, emyF], color= cMapTheme(0),
                    linestyle = '-', lw = 2)

    # # vergence approximation: ray orientation across surface
    # approxLp = round(-1 * (uEmList[0] - uEmList[1]) / (yPosList[0] - yPosList[1]), 3)
    # # print('The approximate emergent vergence is:', approxLp, 'D.')

    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

def drawDevDisp(devThinDeg, prismDiopter, distances):

    fig, ax = plt.subplots(figsize = (6, 4))
    ax.set_aspect('equal', adjustable='box')

    maxDist = max(distances) * 1.2
    if devThinDeg != None:
      devThinRad = math.radians(-devThinDeg)
      maxDisp = math.tan(devThinRad) * maxDist
    elif prismDiopter != None:
      maxDisp = (prismDiopter * maxDist) / -100

    ax.plot([0, maxDist], [0, 0], color = 'black', linewidth = 1, linestyle = '--', label = 'Original Path')
    ax.quiver(0, 0, maxDist, maxDisp, scale_units = 'xy', scale =1 ,
            color = cMapTheme(0), width = 0.005, label= 'Refracted')

    for i, dist in enumerate(distances):

        # compute displacement for the current distance
        if devThinDeg != None:
          disp = round(math.tan(devThinRad) * dist, 3)
        elif prismDiopter != None:
          disp = round(prismDiopter * dist, 3) / -100

        # draw displacement
        ax.plot([dist, dist], [0, disp], color = cMapIndex(i),
                linestyle = '--', linewidth=1.5)

        # add text annotations for dist and disp values
        # custom text bok
        ctbox = dict(facecolor = cMapTheme(i + 1), edgecolor = cMapTheme(0),
            boxstyle = 'round, pad = 0.3', linewidth = 1, alpha = 1)

        labelStrDist = ["$dist_", str(i), "$"]; strCatDist = "".join(labelStrDist)
        ax.text(dist, 0, strCatDist, color=cMapTheme(0),
                ha='center', va='center', zorder=2, bbox=ctbox)

        labelStrDisp = ["$disp_", str(i), "$"]; strCatDisp = "".join(labelStrDisp)
        ax.text(dist, disp, strCatDisp, color=cMapTheme(0),
                ha='center', va='center', zorder=2, bbox=ctbox)

        print('The displacement at', round(dist, 3), 'm is', disp, 'm.')

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Displacement (m)')
    ax.axis((0, maxDist, maxDisp, 1))
    ax.grid(False)
    ax.set_frame_on(False)

    ax.legend(loc='lower left', labelcolor = cMapTheme(7), facecolor= cMapTheme(5),
            edgecolor = cMapTheme(0))
    plt.show()

def drawDualInterface(theta1Deg, n, np, npp, t):
  # convert incident angle to radians
  theta1Rad = math.radians(theta1Deg)

  # setup the plot
  fig, ax = plt.subplots(figsize=(6, 6))
  # dynamic plot limits based on t and a reasonable ray length scale
  plotSpan = max(t, 0.5) * 2
  ax.set_xlim(-plotSpan, plotSpan)
  ax.set_ylim(-plotSpan, plotSpan)
  ax.set_aspect('equal')
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_frame_on(False)

  halfT = t / 2

  # determine colors for the three media based on their refractive indices
  leftMedColor = getInterpolatedColor(n)
  midMedColor = getInterpolatedColor(np)
  rightMedColor = getInterpolatedColor(npp)

  # define the media
  leftRect = patches.Rectangle((-plotSpan, -plotSpan), plotSpan - halfT, 2 * plotSpan, facecolor=leftMedColor,
                              alpha=0.6, zorder=0)
  midRect = patches.Rectangle((-halfT, -plotSpan), t, 2 * plotSpan, facecolor=midMedColor,
                               alpha=0.6, zorder=0)
  rightRect = patches.Rectangle((halfT, -plotSpan), plotSpan - halfT, 2 * plotSpan, facecolor=rightMedColor,
                                 alpha=0.6, zorder=0)

  # draw the media
  ax.add_patch(leftRect); ax.add_patch(midRect); ax.add_patch(rightRect)

  #> plot light direction convention
  ax.quiver(-plotSpan, -plotSpan * 0.95, 2 * plotSpan, 0, color = cMapTheme(0),
             scale_units = 'xy', scale = 1, width = 0.003)

  # draw the first (x = -halfT) and second (x = halfT) surfaces
  ax.axvline(x=-halfT, color=cMapTheme(0), linestyle='-', linewidth = 1, zorder=1)
  ax.axvline(x=halfT, color=cMapTheme(0), linestyle='-', linewidth = 1, zorder=1)

  # draw the normal for the first surface
  normLength = plotSpan * 0.2
  ax.plot([-halfT - normLength/2, -halfT + normLength/2], [0, 0],
          color=cMapTheme(0), linestyle='--', linewidth=1, zorder=2)

  # length of the rays for plotting
  rayLength = plotSpan * 0.7

  #> incident ray
  ix0 = -halfT - rayLength * math.cos(theta1Rad)
  iy0 = rayLength * math.sin(theta1Rad)
  ixF = -halfT
  iyF = 0
  ax.plot([ix0, ixF], [iy0, iyF], color=cMapTheme(0), linewidth=2,
          label='Refractions', zorder=2)

  # initialize final x and y coordinates for original path; default edge of plot
  finalXOrig = plotSpan

  #>>> INTERFACE 1
  thetaP1Deg = None # initialize emergent angle for interface 1
  yAt2 = 0 # y-coordinate where ray hits second interface

  try:
    sinAngle1 = (n * math.sin(theta1Rad)) / np
    if abs(sinAngle1) <= 1:
      thetaP1Rad = math.asin(sinAngle1)
      thetaP1Deg = math.degrees(thetaP1Rad)

      # ray emergent from first interface
      ex10 = -halfT
      ey10 = 0
      yAt2 = -t * math.tan(thetaP1Rad)
      ex1F = halfT
      ey1F = yAt2
      ax.plot([ex10, ex1F], [ey10, ey1F], color=cMapTheme(0), linestyle='-',
              linewidth=2, zorder=2)

      # draw the normal for the second surface
      ax.plot([halfT - normLength/2, halfT + normLength/2], [yAt2, yAt2],
              color=cMapTheme(0), linestyle='--', linewidth=1, zorder=2)

      #>>> INTERFACE 2
      thetaP2Deg = None # initialize emergent angle for interface 2
      try:
        sinAngle2 = (np * math.sin(thetaP1Rad)) / npp
        if abs(sinAngle2) <= 1:
          thetaP2Rad = math.asin(sinAngle2)
          thetaP2Deg = math.degrees(thetaP2Rad)

          # Refracted ray 2 (into medium npp)
          ex20 = halfT
          ey20 = yAt2
          ex2F = halfT + rayLength * math.cos(thetaP2Rad)
          ey2F = yAt2 - rayLength * math.sin(thetaP2Rad)
          ax.plot([ex20, ex2F], [ey20, ey2F], color=cMapTheme(0), linestyle='-',
                  linewidth=2, zorder=2)

          # update the final x so that original path and final emergent ray match
          finalXOrig = ex2F

        else:
          ax.text(plotSpan * 0.9, 0, 'Total Internal Reflection', color=cMapTheme(0), ha='center', va='center',
                  zorder=2, bbox=tbox)
      except ValueError:
        ax.text(plotSpan * 0.9, 0, 'Total Internal Reflection', color=cMapTheme(0), ha='center', va='center',
                zorder=2, bbox=tbox)
    else:
      ax.text(-plotSpan * 0.9, 0, 'Total Internal Reflection', color=cMapTheme(0), ha='center', va='center',
              zorder=2, bbox=tbox)
  except ValueError:
    ax.text(-plotSpan * 0.9, 0, 'Total Internal Reflection', color=cMapTheme(0), ha='center', va='center',
            zorder=2, bbox=tbox)

  # draw original path of light
  finalYOrig = iy0 - (finalXOrig - ix0) * math.tan(theta1Rad)
  ax.plot([ix0, finalXOrig], [iy0, finalYOrig],
          color=cMapTheme(0), linestyle='--', linewidth=1, zorder=1, label='Original Path')

  ax.legend(loc='upper right', labelcolor=cMapTheme(7), facecolor= cMapTheme(5),
            edgecolor=cMapTheme(0))
  plt.show()

  # print statements for angles and reflection proportion
  print(f"θ1: {theta1Deg:.2f} degrees")
  if thetaP1Deg is not None:
    print(f"θ′1: {thetaP1Deg:.2f} degrees")
    print(f"θ2: {thetaP1Deg:.2f} degrees")
    if thetaP2Deg is not None:
      print(f"θ′2: {thetaP2Deg:.2f} degrees")

      # only compute and show displacement for parallel blocks
      if n == npp:
        d = computeDisplacement(n, theta1Deg, np, t) # compute lateral displacement
        print(f"Lateral Displacement: {d:.4f} mm")
    else:
      print("Total Internal Reflection at second interface.")
  else:
    print("Total Internal Reflection at first interface.")

def drawPinhole(h, hp, l, lp):
  #> inputs
  # h, hp, l, and lp are the object and image heights/distances

  # create a figure and a set of subplots
  fig, ax = plt.subplots(figsize = (6, 4))

  #> plot light direction convention
  ax.quiver(xLim[0], yLim[0] * 0.95, xLim[1] - xLim[0], 0, color = cMapTheme(0), scale_units = 'xy',
            scale = 1, width = 0.003, zorder = 2)

  #> plot optical axis
  ax.plot([xLim[0], xLim[1]], [0, 0], color = cMapTheme(0), linewidth = 0.5, zorder = 0)

  # solving for object height
  if h == None:
    h = (hp * l) / lp
    answer = h

    print("An image with a height of", hp, "m is formed", lp, "m from a pinhole.")
    print("The object is placed", l, "m from the pinhole.")
    print("How large is the object?")

  # solving for image height
  elif hp == None:
    hp = (h * lp) / l
    answer = hp

    print("An object with a height of", h, "m, is placed", l, "m from a pinhole.")
    print("The screen is placed", lp, "m from the pinhole.")
    print("How large is the image?")

  # solving for object distance
  elif l == None:
    l = (h * lp) / hp
    answer = l

    print("An image with a height of", hp, "m is formed", lp, "m from a pinhole.")
    print("The object is", h, "m tall.")
    print("Where is the object?")

  # solving for image distance
  elif lp == None:
    lp = (hp * l) / h
    answer = lp

    print("An object with a height of", h, "m, is placed", l, "m from a pinhole.")
    print("An image with a height of", hp, "m is formed.")
    print("Where is the image?")

  #> draw pinhole
  ax.scatter(0, 0, c = cMapTheme(0), s = 100)
  ax.scatter(0, 0, c = cMapTheme(7), s = 50)
  # ax.text(0, textY, 'Pinhole', color = cMapTheme(7), ha = 'center', bbox = tbox)

  #> draw screen
  ax.plot([lp, lp], [yLim[0], yLim[1]], color = (0.80, 0.80, 0.80), linewidth = 5, zorder = 0)

  #> draw object
  ax.quiver(l, 0, 0, h, color = cMapTheme(0),
          scale_units = 'xy', scale = 1, width = 0.0025)
  ax.text(l, textY, 'RO', color = cMapTheme(7), ha = 'center', bbox = tbox)

  #> draw image
  ax.quiver(lp, 0, 0, hp, color = cMapTheme(0),
            scale_units = 'xy', scale = 1, width = 0.0025)
  ax.text(lp, textY, 'RI', color = cMapTheme(7), ha = 'center', bbox = tbox)

  #> draw rays
  ax.quiver(l, h, lp-l, hp-h, color = cMapTheme(0),
            scale_units = 'xy', scale = 1, width = 0.001)
  ax.quiver(l, 0, lp-l, 0, color = cMapTheme(0),
            scale_units = 'xy', scale = 1, width = 0.001)

  # set plot details and display
  ax.set_aspect('equal')
  ax.set_xlim(xLim)
  ax.set_ylim(yLim)
  ax.axis('off')
  plt.show()

  return round(answer, 3)

def drawPlaneInterface(n, l, np, lp, obsDist, emPos):

  # solve for unknown
  answer = None # Initialize answer
  if n is None:
    n = (np * l) / lp
    answer = round(n, 3)
    print('The incident refractive index is', n, '.')
  elif l is None:
    l = (n * lp) / np
    answer = round(l, 3)
    print('The object distance is', l, 'm.')
  elif np is None:
    np = (n * lp) / l
    answer = round(np, 3)
    print('The emergent refractive index is', np, '.')
  elif lp is None:
    lp = round((np * l) / n, 3)
    print('The image distance is', lp, 'm.')
    answer = lp

  L = round(n/l, 3)
  Lp = round(np/lp, 3)

  # determine dynamic axis limits based on l and lp
  maxCoord = max(abs(l) if l is not None else 0, abs(lp) if lp is not None else 0, abs(obsDist), 0.5) # Ensure a minimum value for scaling
  padding = 0.2 * maxCoord
  plotLimit = maxCoord + padding

  # setup the plot with dynamic limits
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.set_xlim(-plotLimit, plotLimit)
  ax.set_ylim(-plotLimit, plotLimit)
  ax.set_aspect('equal')
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_frame_on(False)

  # plot incident medium, object location, emergent medium and image location

  # determine colors for n and np media 
  nCol = getInterpolatedColor(n)
  nColp = getInterpolatedColor(np)

  # draw media rectangles and light direction arrow based on emPos
  if emPos == 'bottom':
    # top half is incident medium (n), bottom half is emergent medium (np)
    iRect = patches.Rectangle((-plotLimit, 0), 2 * plotLimit, plotLimit, facecolor=nCol, alpha=0.6, zorder=0)
    eRect = patches.Rectangle((-plotLimit, -plotLimit), 2 * plotLimit, plotLimit, facecolor=nColp, alpha=0.6, zorder=0)
    # light direction arrow points downwards
    ax.quiver(-plotLimit * 0.9, plotLimit * 0.9, 0, -1.8 * plotLimit * 0.9 / 0.9, color=cMapTheme(0), scale_units='xy', scale=1, width=0.003 , zorder=2)
    ax.text(0, -obsDist, 'Observer', ha='center', va='center',
            color = cMapTheme(7), bbox = tbox)

  elif emPos == 'top':
    # top half is emergent medium (np), bottom half is incident medium (n)
    iRect = patches.Rectangle((-plotLimit, -plotLimit), 2 * plotLimit, plotLimit, facecolor=nCol, alpha=0.6, zorder=0)
    eRect = patches.Rectangle((-plotLimit, 0), 2 * plotLimit, plotLimit, facecolor=nColp, alpha=0.6, zorder=0)
    # light direction arrow points upwards
    ax.quiver(-plotLimit * 0.9, -plotLimit * 0.9, 0, 1.8 * plotLimit * 0.9 / 0.9, color=cMapTheme(0), scale_units='xy', scale=1, width=0.003, zorder=2)
    ax.text(0, obsDist, 'Observer', ha='center', va='center',
            color = cMapTheme(7), bbox = tbox)

  else:
    raise ValueError("emPos must be 'top' or 'bottom'")

  ax.add_patch(iRect)
  ax.add_patch(eRect)

  # draw the surface (x-axis)
  ax.axhline(y=0, color=cMapTheme(0), linestyle='-', linewidth = 1, zorder=1)

  # Plot object and image locations
  # Using small squares for illustration
  oSize = plotLimit * 0.1 / 1 # Scale object size relative to plotLimit
  iSize = plotLimit * 0.1 / 1 # Scale image size relative to plotLimit

  # determine plotting y-coordinates for object and image to ensure object is in incident medium
  obPlotY = l
  if emPos == 'bottom':
      # incident medium is y > 0
      obPlotY = abs(l)
      imPlotY = abs(lp)
  elif emPos == 'top':
      # incident medium is y < 0
      obPlotY = l;
      imPlotY = lp;

  # object location (x=0, y=obPlotY)
  object_patch = patches.Rectangle((0 - oSize/2, obPlotY - oSize/2), oSize, oSize,
    facecolor=cMapTheme(3), edgecolor=cMapTheme(0), zorder=3, label='Object')
  ax.add_patch(object_patch)

  # image location (x=0, y=imPlotY)
  image_patch = patches.Rectangle((0 - iSize/2, imPlotY - iSize/2), iSize, iSize,
    alpha = 0.5, facecolor=cMapTheme(4), edgecolor=cMapTheme(0), zorder=3, label='Image')
  ax.add_patch(image_patch)

  ax.legend(loc='upper right', labelcolor=cMapTheme(7), facecolor= cMapTheme(5),
            edgecolor=cMapTheme(0))
  plt.show()

  print('The image is located', round(abs(lp) + obsDist, 3), 'm from the observer.')

def drawPrismCombo(prismA, typeA, prismB, typeB, eye):

    # convert all vectors to cartesian coords for vector addition
    def toCartesian(coords, vType):
        if vType == 'P':
            mag, angleDeg = coords
            # Normalize angle to be between 0 and 360 degrees for consistent interpretation
            angleDeg = (angleDeg + 360) % 360
            angleRad = numpy.radians(angleDeg)
            x = mag * math.cos(angleRad)
            y = mag * math.sin(angleRad)
            return x, y
        elif vType == 'C':
            return coords
        else:
            raise ValueError("Vector type must be 'C' or 'P'.")

    # cartesian coordinates
    x1, y1 = toCartesian(prismA, typeA)
    x2, y2 = toCartesian(prismB, typeB)

    # polar coordinates
    magA = math.sqrt(x1**2 + y1**2)
    dirARad = math.atan2(y1, x1)
    dirADeg = numpy.degrees(dirARad)

    magB = math.sqrt(x2**2 + y2**2)
    dirBRad = math.atan2(y2, x2)
    dirBDeg = numpy.degrees(dirBRad)

    # resultants
    rx = x1 + x2
    ry = y1 + y2

    # polar coordinates for resultant
    rMag = math.sqrt(rx**2 + ry**2)
    rDirRad = math.atan2(ry, rx)
    rDirDeg = numpy.degrees(rDirRad)
    # Ensure resultant direction is between 0 and 360 degrees for display
    rDirDeg = (rDirDeg + 360) % 360

    # define colormap and helper function
    cmap = plt.get_cmap('viridis') # Viridis colormap
    def getColorFromAng(angle_deg):
        # Normalize angle from [-180, 180] to [0, 1] for colormap
        normAng = (angle_deg + 360) % 360 # Ensure angle is [0, 360)
        return cmap(normAng / 360.0)

    colorA = getColorFromAng(dirADeg)
    colorB = getColorFromAng(dirBDeg)
    colorR = getColorFromAng(rDirDeg)

    #> PLOTTING
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_aspect('equal', adjustable='box')

    # individual vectors
    ax.plot([0, x1], [0, y1], color=colorA, lw = 2)
    ax.plot([0, x2], [0, y2], color=colorB, lw = 2)

    # resultant vector
    ax.plot([0, rx], [0, ry], color = colorR, lw = 3)

    # Add text labels at the tips of the vectors
    ax.text(x1, y1, 'Base A', color=colorA, ha='center', va='bottom', fontsize=10)
    ax.text(x2, y2, 'Base B', color=colorB, ha='center', va='bottom', fontsize=10)
    ax.text(rx, ry, 'R', color=colorR, ha='center', va='bottom', fontsize=12, fontweight='bold')

    # dynamic plot limits
    maxVal = max(abs(x1), abs(y1), abs(x2), abs(y2), abs(rx), abs(ry), rMag) * 1.2
    ax.set_xlim(-maxVal, maxVal)
    ax.set_ylim(-maxVal, maxVal)

    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # get plot limits for custom labels
    minX, maxX = ax.get_xlim()
    minY, maxY = ax.get_ylim()

    # lines orthogonal to vectors to represent prism bases
    baseLength = maxVal * 0.08 # Length of the orthogonal line segment

    def drawBase(axObj, vx, vy, color):
        if vx == 0 and vy == 0:
            return
        mag = math.sqrt(vx**2 + vy**2)
        if mag == 0:
            return
        npx = -vy / mag
        npy = vx / mag
        startX = vx - npx * (baseLength/2)
        startY = vy - npy * (baseLength/2)
        endX = vx + npx * (baseLength/2)
        endY = vy + npy * (baseLength/2)
        axObj.plot([startX, endX], [startY, endY], color=color,
                   linestyle = '-', linewidth = 2)

    drawBase(ax, x1, y1, colorA)
    drawBase(ax, x2, y2, colorB)
    drawBase(ax, rx, ry, colorR)

    # draw eye
    irisRad = maxVal * 0.05
    pupilRad = maxVal * 0.025

    iris = patches.Circle((0, 0), irisRad, color = (0, 0.60, 0.80), zorder = 3)
    pupil = patches.Circle((0, 0), pupilRad, color= cMapTheme(0), zorder = 4)
    ax.add_patch(iris)
    ax.add_patch(pupil)

    # draw nose
    noseXOffset = maxVal * 0.12
    noseYOffset = -maxVal * 0.15

    ctbox = dict(facecolor = cMapTheme(5), edgecolor = cMapTheme(0),
                boxstyle = 'round, pad = 0.3', linewidth = 1, alpha = 0.60)

    if eye == 'OD':
        noseCoords = [(noseXOffset - maxVal * 0.03, noseYOffset),
            (noseXOffset, noseYOffset - maxVal * 0.045),
            (noseXOffset + maxVal * 0.03, noseYOffset)]
        ax.text(minX + 0.05 * (maxX - minX), 0, "RIGHT/OUT", ha='left', va='center', fontsize=12, color=cMapTheme(0), bbox=ctbox)
        ax.text(maxX - 0.05 * (maxX - minX), 0, "LEFT/IN", ha='right', va='center', fontsize=12, color=cMapTheme(0), bbox=ctbox)
        ax.text(0, maxY - 0.05 * (maxY - minY), "UP", ha='center', va='top', fontsize=12, color=cMapTheme(0), bbox=ctbox)
        ax.text(0, minY + 0.05 * (maxY - minY), "DOWN", ha='center', va='bottom', fontsize=12, color=cMapTheme(0), bbox=ctbox)

    elif eye == 'OS':
        noseCoords = [(-noseXOffset + maxVal * 0.03, noseYOffset),
            (-noseXOffset, noseYOffset - maxVal * 0.045),
            (-noseXOffset - maxVal * 0.03, noseYOffset)]
        ax.text(minX + 0.05 * (maxX - minX), 0, "RIGHT/IN", ha='left', va='center', fontsize=12, color=cMapTheme(0), bbox=ctbox)
        ax.text(maxX - 0.05 * (maxX - minY), 0, "LEFT/OUT", ha='right', va='center', fontsize=12, color=cMapTheme(0), bbox=ctbox)
        ax.text(0, maxY - 0.05 * (maxY - minY), "UP", ha='center', va='top', fontsize=12, color=cMapTheme(0), bbox=ctbox)
        ax.text(0, minY + 0.05 * (maxY - minY), "DOWN", ha='center', va='bottom', fontsize=12, color=cMapTheme(0), bbox=ctbox)
    else:
        raise ValueError("Eye must be 'OD' or 'OS'.")

    nose = patches.Polygon(noseCoords, closed = True, color= cMapTheme(0), zorder = 2)
    ax.add_patch(nose)

    plt.show()

    arx = abs(rx)
    ary = abs(ry)
    if eye == 'OD':
      if rx > 0:
        if ry > 0:
          print(f"Cartesian: {arx:.2f} ∆ BI, {ary:.2f} ∆ BU")
        elif ry < 0:
          print(f"Cartesian: {arx:.2f} ∆ BI, {ary:.2f} ∆ BD")
      elif rx < 0:
        if ry > 0:
          print(f"Cartesian: {arx:.2f} ∆ BO, {ary:.2f} ∆ BU")
        elif ry < 0:
          print(f"Cartesian: {arx:.2f} ∆ BO, {ary:.2f} ∆ BD")
    elif eye == 'OS':
      if rx > 0:
        if ry > 0:
          print(f"Cartesian: {arx:.2f} ∆ BO, {ary:.2f} ∆ BU")
        elif ry < 0:
          print(f"Cartesian: {arx:.2f} ∆ BO, {ary:.2f} ∆ BD")
      elif rx < 0:
        if ry > 0:
          print(f"Cartesian: {arx:.2f} ∆ BI, {ary:.2f} ∆ BU")
        elif ry < 0:
          print(f"Cartesian: {arx:.2f} ∆ BI, {ary:.2f} ∆ BD")

    print(f"Polar: {rMag:.2f} ∆ @ {rDirDeg:.2f}°")

def drawRefraction(n, np, givens, givenVals, randBool, unknowns):

  # givens
  if givens[0] == 'l': incSwitch = 0
  if givens[0] == 'L': incSwitch = 1
  if givens[0] == 'ri': incSwitch = 2
  if givens[0] == 'Vi': incSwitch = 3

  if givens[1] == 'F': powSwitch = 0
  if givens[1] == 'f': powSwitch = 1
  if givens[1]  == 'fp': powSwitch = 2
  if givens[1] == 'r': powSwitch = 3

  if givens[2] == 'lp': emSwitch = 0
  if givens[2] == 'Lp': emSwitch = 1
  if givens[2] == 're': emSwitch = 2
  if givens[2] == 'Ve': emSwitch = 3

  # unknowns
  if unknowns[0] == 'l': incSwitch = 4
  if unknowns[0] == 'L': incSwitch = 5
  if unknowns[0] == 'ri': incSwitch = 6
  if unknowns[0] == 'Vi': incSwitch = 7

  if unknowns[1] == 'F': powSwitch = 4
  if unknowns[1] == 'f': powSwitch = 5
  if unknowns[1] == 'fp': powSwitch = 6
  if unknowns[1] == 'r': powSwitch = 7

  if unknowns[2] == 'lp': emSwitch = 4
  if unknowns[2] == 'Lp': emSwitch = 5
  if unknowns[2] == 're': emSwitch = 6
  if unknowns[2] == 'Ve': emSwitch = 7

  # set to none to ensure proper assignment
  l = None
  F = None
  lp = None

  # create a figure and a set of subplots
  fig, ax = plt.subplots(figsize=(6, 5))

  llpVergList = [V for V in range(-20, 20) if V not in range(0, 1)]
  iVergList = [V for V in range(-20, 20) if V not in range(-4, 1)]
  eVergList = [V for V in range(-20, 20) if V not in range(-1, 4)]

  # givens for incident light
  # l
  if incSwitch == 0:
    if randBool == True:
      l = round(npr.uniform(-1, 1), 3)
    else:
      l = givenVals[0]
    L = round(safeDivide(n, l), 3)
    print('The object distance is:', l, 'm.')

  # L
  if incSwitch == 1:
    if randBool == True:
      L = round(npr.choice(llpVergList), 3)
    else:
      L = givenVals[0]
    l = round(safeDivide(n, L), 3)
    print('The incident vergence is:', L, 'D.')

  # [ri, di]
  elif incSwitch == 2:
    if randBool == True:
      ri = round(npr.uniform(-0.25, 1), 3)
      Vi = round(safeDivide(n, ri), 3)
      di = round(npr.uniform(-0.75, 0), 3)
    else:
      ri = givenVals[0][0]
      di = givenVals[0][1]
      Vi = round(safeDivide(n, ri), 3)
    l = di + ri
    L = round(safeDivide(n, l), 3)
    print('An incident wavefront with a radius of', ri, 'm is located', di,
          'm left of the refractive interface.')
  # [Vi, di]
  elif incSwitch == 3:
    if randBool == True:
      Vi = round(npr.choice(iVergList), 3)
      ri = round(safeDivide(n, Vi), 3)
      di = round(npr.uniform(-0.75, 0), 3)
    else:
      Vi = givenVals[0][0]
      di = givenVals[0][1]
      ri = round(safeDivide(n, Vi), 3)
    l = di + ri
    L = round(safeDivide(n, l), 3)
    print('An incident wavefront with a vergence of', Vi, 'D is located', di,
        'm left of the refractive interface.')

  # givens for power
  # F
  if powSwitch == 0:
    if randBool == True:
      F = round(npr.choice(llpVergList), 3)
    else:
      F = givenVals[1]
    f = round(safeDivide(-n, F), 3)
    fp = round(safeDivide(np, F), 3)
    print('The power of the refractive interface is', F, 'D.')

  if powSwitch == 1:
    if randBool == True:
      F = round(npr.choice(llpVergList), 3)
      f = round(safeDivide(-n, F), 3)
    else:
      f = givenVals[1]
      F = round(safeDivide(-n, f), 3)
    fp = round(safeDivide(np, F), 3)
    print('The primary focal length is', f, 'm.')

  if powSwitch == 2:
    if randBool == True:
      F = round(npr.choice(llpVergList), 3)
      fp = round(safeDivide(np, F), 3)
    else:
      fp = givenVals[1]
      F = round(safeDivide(np, fp), 3)
    f = round(safeDivide(-n, F), 3)
    print('The secondary focal length is', fp, 'm.')

  if powSwitch == 3:
    if randBool == True:
      F = round(npr.choice(llpVergList), 3)
      r = round(safeDivide(np - n, F), 3)
    else:
      r = givenVals[1]
      F = round(safeDivide(np - n, r), 3)
    f = round(safeDivide(-n, F), 3)
    fp = round(safeDivide(np, F), 3)
    print('The radius of curvature is', r, 'm.')

  # givens for emergent light
  # lp
  if emSwitch == 0:
    if randBool == True:
      lp = round(npr.uniform(-1, 1), 3)
    else:
      lp = givenVals[2]
    Lp = round(safeDivide(np, lp), 3)
    print('The image distance is:', lp, 'm.')

  # Lp
  elif emSwitch == 1:
    if randBool == True:
      Lp = round(npr.choice(llpVergList), 3)
    else:
      Lp = givenVals[2]
    lp = round(safeDivide(np, Lp), 3)
    print('The emergent vergence is:', Lp, 'D.')

  # [re, de]
  elif emSwitch == 2:
    if randBool == True:
      re = round(npr.uniform(-1, 0.25), 3)
      Ve = round(safeDivide(np, re), 3)
      de = round(npr.uniform(0, 0.75), 3)
    else:
      re = givenVals[2][0]
      de = givenVals[2][1]
      Ve = round(safeDivide(np, re), 3)
    lp = de + re
    Lp = round(safeDivide(np, lp), 3)
    print('An emergent wavefront with a radius of', re, 'm is located', de,
          'm right of the refractive interface.')
  # [Ve, de]
  elif emSwitch == 3:
    if randBool == True:
      Ve = round(npr.choice(eVergList), 3)
      re = round(safeDivide(n, Ve), 3)
      de = round(npr.uniform(0, 0.75), 3)
    else:
      Ve = givenVals[2][0]
      de = givenVals[2][1]
      re = round(safeDivide(np, Ve), 3)
    lp = de + re
    Lp = round(safeDivide(np, lp), 3)
    print('An emergent wavefront with a vergence of', Ve, 'D is located', de,
          'm right of the refractive interface.')

  givensBool = sum([l != None, F != None, lp != None])
  if givensBool < 2:
    raise ValueError('You have not provided enough givens.')
  elif givensBool > 2:
    raise ValueError('You have chosen too many givens.')

  #> unknowns for incident light
  # solving for a parameter related to incident light
  # assume lp/LP and F are defined

  # l
  if incSwitch == 4:
    print('What is the object distance?')
    L = round(Lp - F, 3)
    l = round(safeDivide(n, L), 3)
    answer = l
  # L
  elif incSwitch == 5:
    print('What is the vergence incident at the refractive interface?')
    L = round(Lp - F, 3)
    l = round(safeDivide(n, L), 3)
    answer = L

  # [ri, di]
  elif incSwitch == 6:
    di = round(npr.uniform(-0.75, 0), 3)
    print('What is the radius of an incident wavefront located', di,
          'm left of the refractive interface?')
    L = round(Lp - F, 3)
    l = round(safeDivide(n, L), 3)
    ri = l - di
    answer = ri

  # [Vi, di]
  elif incSwitch == 7:
    di = round(npr.uniform(-0.75, 0), 3)
    print('What is the vergence of an incident wavefront located', di,
          'm left of the refractive interface?')
    L = round(Lp - F, 3)
    l = round(safeDivide(n, L), 3)
    ri = l - di
    Vi = round(safeDivide(n, ri), 3)
    answer = Vi

  #> unknowns for power
  # solving for a parameter related to power
  # assume l/L and lp/LP are defined
  # F / f / fp
  if powSwitch >= 4:
    F = round(Lp - L, 3)
    f = round(safeDivide(-n, F), 3)
    fp = round(safeDivide(np, F), 3)
    r = round(safeDivide(np - n, F), 3)

  if powSwitch == 4:
    print('What is the power of the refractive interface?')
    answer = F
  # f  
  elif powSwitch == 5:
    print('What is the primary focal length of the refractive interface?')
    answer = f
  # fp
  elif powSwitch == 6:
    print('What is the secondary focal length of the refractive interface?')
    answer = fp
  # r
  elif powSwitch == 7:
    print('What is the radius of curvature of the refractive interface?')
    answer = r

  #> unknowns for emergent light
  # solving for a parameter related to incident light
  # assume l/L and F are defined
  # lp
  if emSwitch == 4:
    print('What is the image distance?')
    Lp = round(L + F, 3)
    lp = round(safeDivide(np, Lp), 3)
    answer = lp
  # Lp
  elif emSwitch == 5:
    print('What is the vergence emergent from the refractive interface?')
    Lp = round(L + F, 3)
    lp = round(safeDivide(np, Lp), 3)
    answer = Lp
  # re
  elif emSwitch == 6:
    de = round(npr.uniform(0, 0.75), 3)
    print('What is the radius of an emergent wavefront located', de,
          'm right of the refractive interface?')
    Lp = round(L +  F, 3)
    lp = round(safeDivide(np, Lp), 3)
    re = lp - de
    answer = re
  # Ve
  elif emSwitch == 7:
    de = round(npr.uniform(0, 0.75), 3)
    print('What is the vergence of an emergent wavefront located', de,
          'm right of the refractive interface?')
    Lp = round(L + F, 3)
    lp = round(safeDivide(np, Lp), 3)
    re = lp - de
    Ve = round(safeDivide(np, re), 3)
    answer = Ve

  # r = round((np - n) / F, 3)
  # print('The radius of the surface is', r, 'm.')
  LM = round(safeDivide(L, Lp), 3)
  # print('The lateral magnification is', LM, '.')

  #> draw object
  ax.scatter(l, 0, c = cMapTheme(0), s = 20)
  if l > 0: # virtual object
    ax.text(l, textY, 'VO', color = cMapTheme(7), ha = 'center', bbox = tbox)
  elif l < 0: # real object
    ax.text(l, textY, 'RO', color = cMapTheme(7), ha = 'center', bbox = tbox)

  # draw incident rays
  if L > 0: # virtual object
    x = numpy.linspace(xLim[0], l, 101)
    y1 = numpy.linspace(yLim[0] * 0.80, 0, 101)
    y2 = numpy.linspace(yLim[1] * 0.80, 0, 101)

  elif L < 0: # real object
    x = numpy.linspace(l, 0, 101)
    y1 = numpy.linspace(0, yLim[0] * 0.80, 101)
    y2 = numpy.linspace(0, yLim[1] * 0.80, 101) 

  elif L == 0: # parallel incidence
    x = numpy.linspace(xLim[0], 0, 101)
    y1 = numpy.linspace(yLim[0] * 0.80, yLim[0] * 0.80, 101)
    y2 = numpy.linspace(yLim[1] * 0.80, yLim[1] * 0.80, 101)

  # find intercepts at surface; this is where emergent rays will begin
  mi1 = (y1[-1] - y1[0])/(x[-1] - x[0])
  yi1 = y1[0] - mi1 * x[0]
  mi2 = (y2[-1] - y2[0])/(x[-1] - x[0])
  yi2 = y2[0] - mi2 * x[0]

  x[x > 0] = numpy.nan # eliminate positive xCoords for incident rays

  ax.plot(x, y1, color = cMapTheme(0), linewidth = 0.5)
  ax.plot(x, y2, color = cMapTheme(0), linewidth = 0.5)

  #> draw interface
  ax.plot([0, 0], [0, 0.75], color = cMapTheme(0), linewidth = 1)
  ax.plot([0, 0], [0, -0.75], color = cMapTheme(0), linewidth = 1)
  # ax.text(0, -1 * textY, 'SSI', color = cMapTheme(7), ha = 'center', bbox = tbox)

  # draw focal points
  ax.scatter(f, 0, s = 20, facecolors = 'none', edgecolors = cMapTheme(0))
  ax.scatter(fp, 0, s = 20, facecolors = 'none', edgecolors = cMapTheme(0))

  #> draw image
  ax.scatter(lp, 0, c = cMapTheme(0), s = 20)
  if lp < 0: # virtual image
    ax.text(lp, textY, 'VI', color = cMapTheme(7), ha = 'center', bbox = tbox)
  elif lp > 0: # real image
    ax.text(lp, textY, 'RI', color = cMapTheme(7), ha = 'center', bbox = tbox)

  # draw emergent rays
  if Lp < 0: # virtual image
    xTarget = [lp, 0]
    yTarget1 = [0, yi1]
    me1 = (yTarget1[1] - yTarget1[0]) / (xTarget[1] - xTarget[0])
    yInt1 = yTarget1[1] - me1 * xTarget[1]

    yTarget2 = [0, yi2]
    me2 = (yTarget2[1] - yTarget2[0]) / (xTarget[1] - xTarget[0])
    yInt2 = yTarget2[1] - me2 * xTarget[1]

    x = numpy.linspace(lp, xLim[1], 101);
    y1 = me1 * x + yInt1
    y2 = me2 * x + yInt2

  elif Lp > 0: # real image
    xTarget = [0, lp]
    yTarget1 = [yi1, 0]
    me1 = (yTarget1[1] - yTarget1[0]) / (xTarget[1] - xTarget[0])
    yInt1 = yTarget1[1] - me1 * xTarget[1]

    yTarget2 = [yi2, 0]
    me2 = (yTarget2[1] - yTarget2[0]) / (xTarget[1] - xTarget[0])
    yInt2 = yTarget2[1] - me2 * xTarget[1]

    x = numpy.linspace(0, lp, 101)
    y1 = me1 * x + yInt1
    y2 = me2 * x + yInt2

  elif Lp == 0: # parallel emergence
    xTarget = [0, xLim[1]]
    yTarget1 = [yi1, yi1]
    me1 = (yTarget1[1] - yTarget1[0]) / (xTarget[1] - xTarget[0])
    yInt1 = yTarget1[1] - me1 * xTarget[1]

    yTarget2 = [yi2, yi2]
    me2 = (yTarget2[1] - yTarget2[0]) / (xTarget[1] - xTarget[0])
    yInt2 = yTarget2[1] - me2 * xTarget[1]

    x = numpy.linspace(0, xLim[1], 101)
    y1 = numpy.linspace(yLim[0] * 0.80, yLim[0] * 0.80, 101)
    y2 = numpy.linspace(yLim[1] * 0.80, yLim[1] * 0.80, 101)

  x[x < 0] = numpy.nan # eliminate negative xCoords for emergent rays
  ax.plot(x, y1, color = cMapTheme(0), linewidth = 0.5)
  ax.plot(x, y2, color = cMapTheme(0), linewidth = 0.5)

  # set plot details and display
  # calculate colors for incident and emergent sides based on refractive indices
  incColor = getInterpolatedColor(n)
  emColor = getInterpolatedColor(np)

  # background shading for incident and emergent media
  ax.axvspan(xLim[0], 0, facecolor = incColor, alpha=1.0, zorder=-1)
  ax.axvspan(0, xLim[1], facecolor = emColor, alpha=1.0, zorder=-1)

  #> plot light direction convention
  ax.quiver(xLim[0], yLim[0] * 0.95, 2 * xLim[0], 0, color = cMapTheme(3),
             scale_units = 'xy', scale = 1, width = 0.003)

  #> plot optical axis
  ax.plot([xLim[0], xLim[1]], [0, 0], color = cMapTheme(0), linewidth = 0.5,
           zorder = 0)

  ax.set_aspect('equal')
  ax.set_xlim(xLim)
  ax.set_ylim(yLim)
  ax.axis('off')
  plt.show()

  return round(answer, 3)

def drawShadow(h1, h2, d1, d2):
  #> inputs
  # h1, h2, d1, and d2 are the object and shadow heights/distances

  # Create a figure and a set of subplots
  fig, ax = plt.subplots(figsize = (6, 4))

  #> plot light direction convention
  ax.quiver(xLim[0], yLim[0] * 0.95, xLim[1] - xLim[0], 0, color = cMapTheme(3),
             scale_units = 'xy', scale = 1, width = 0.003)

  #> plot optical axis
  ax.plot([xLim[0], xLim[1]], [0, 0], color = cMapTheme(0), linewidth = 0.5,
           zorder = 0)

  # solving for object height
  if h1 == None:
    h1 = (h2 * d1) / d2
    answer = h1

    print("An shadow with a height of", h2, "m is formed", d2, "m from a point source.")
    print("The marble is placed", d1, "m from the point source.")
    print("How large is the marble?")

  # solving for shadow height
  elif h2 == None:
    h2 = (h1 * d2) / d1
    answer = h2

    print("A marble with a diameter of", h1, "m, is placed", d1, "m to the right of a point source.")
    print("The screen is placed", d2, "m from the point source.")
    print("How large is the shadow?")


  # solving for source to object distance
  elif d1 == None:
    d1 = (h1 * d2) / h2
    answer = d1

    print("A shadow with a height of", h2, "m is formed", d2, "m from a point source.")
    print("The marble has a diameter of", h1, "m.")
    print("Where is the marble?")

  # solving for source to shadow distance
  elif d2 == None:
    d2 = (h2 * d1) / h1
    answer = d2

    print("A marble with a diameter of", h1, "m, is placed", d1, "m from a point source.")
    print("A shadow with a height of", h2, "m is formed.")
    print("Where is the screen?")

  #> draw point source
  ax.scatter(-d1, 0, c = cMapTheme(0), s = 25)
  ax.text(-d1, textY, 'RO', color = cMapTheme(7), ha = 'center', bbox = tbox)

  #> draw opaque object
  ax.quiver(0, 0, 0, h1/2, color = cMapTheme(0),
            scale_units = 'xy', angles = 'xy', scale = 1, width = 0.0025)
  ax.quiver(0, 0, 0, -1 * h1/2, color = cMapTheme(0),
            scale_units = 'xy', angles = 'xy', scale = 1, width = 0.0025)
  # ax.text(0, textY, 'Marble', color = cMapTheme(7), ha = 'center', bbox = tbox)

  #> draw wall
  ax.plot([d2 - d1, d2 - d1], [yLim[0], yLim[1]], color = (0.80, 0.80, 0.80), linewidth = 5, zorder = 0)

  #> draw shadow
  ax.quiver(d2 - d1, 0, 0, h2/2, color = (0.4, 0.4, 0.4),
            scale_units = 'xy', scale = 1, width = 0.0025)
  ax.quiver(d2 - d1, 0, 0, -1 * h2/2, color = (0.4, 0.4, 0.4),
            scale_units = 'xy', scale = 1, width = 0.0025)
  ax.text(d2 - d1, textY, 'Shadow', color = cMapTheme(7), ha = 'center',
           bbox = tbox)

  #> draw rays
  ax.quiver(-d1, 0, d1, h2/10, color = cMapTheme(0),
            scale_units = 'xy', scale = 1, width = 0.001)
  ax.quiver(-d1, 0, d2, h2/2, color = cMapTheme(0),
            scale_units = 'xy', scale = 1, width = 0.001)
  ax.quiver(-d1, 0, d2, -1 * h2/2, color = cMapTheme(0),
            scale_units = 'xy', scale = 1, width = 0.001)
  ax.quiver(-d1, 0, d1, -1 * h2/10, color = cMapTheme(0),
            scale_units = 'xy', angles = 'xy', scale = 1, width = 0.001)

  # set plot details and display
  ax.set_aspect('equal')
  ax.set_xlim(xLim)
  ax.set_ylim(yLim)
  ax.axis('off')
  plt.show()

  return round(answer, 3)

def drawSnellInterface(theta, n, np):
  # Convert angle to radians
  thetaRad = math.radians(theta)

  # Setup the plot
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.set_xlim(-1, 1)
  ax.set_ylim(-1, 1)
  ax.set_aspect('equal')
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_frame_on(False)

  topMedColor = getInterpolatedColor(n)
  bottomMedColor = getInterpolatedColor(np)

  #> plot light direction convention
  ax.quiver(-0.9, 0.9, 0, -1.8, color = cMapTheme(0), scale_units = 'xy',
            scale = 1, width = 0.003, zorder = 2)

  # draw the top medium (incident medium)
  topRect = patches.Rectangle((-1, 0), 2, 1, facecolor=topMedColor,
                              alpha=0.6, zorder=0)
  ax.add_patch(topRect)

  # draw the bottom medium (emergent medium)
  bottomRect = patches.Rectangle((-1, -1), 2, 1, facecolor=bottomMedColor,
                                 alpha=0.6, zorder=0)
  ax.add_patch(bottomRect)

  # # Draw the surface (x-axis)
  # ax.axhline(y=0, color=cMapTheme(0), linestyle='-', linewidth = 1, zorder=1)

  # Draw the normal (y-axis)
  ax.axvline(x=0, color=cMapTheme(0), linestyle='-', linewidth = 0.5, zorder=1)

  # length of the rays for plotting purposes
  rayLength = 1

  # calculate proportion of reflected light
  # fresnel's law adapted for general use
  R = ((np - n) / (np + n))**2

  # Define linewidths based on reflection proportion
  minLineWidth = 1.5 # Minimum linewidth
  maxLwJump = 3 # Maximum increase over minLineWidth

  reflected_lw = minLineWidth + R * maxLwJump
  # For refracted light, the proportion is (1 - R)
  refracted_lw = minLineWidth + (1 - R) * maxLwJump

  # Incident ray: starts from top-left/right and goes to the origin (0,0)
  ixF = 0
  iyF = 0

  ix0 = -rayLength * math.sin(thetaRad)
  iy0 = rayLength * math.cos(thetaRad)

  # incident rays
  rayAdj = 0.2
  ax.plot([ix0, ixF], [iy0, iyF], color=cMapTheme(0), linewidth=2, label='Incident', zorder=2)
  ax.plot([ix0 + rayAdj, ixF + rayAdj], [iy0, iyF], color=cMapTheme(0), linewidth=0.25, zorder=2)
  ax.plot([ix0 - rayAdj, ixF - rayAdj], [iy0, iyF], color=cMapTheme(0), linewidth=0.25, zorder=2)

  # reflected ray
  rx0 = 0
  ry0 = 0

  rxF = rayLength * math.sin(thetaRad)
  ryF = rayLength * math.cos(thetaRad)

  ax.plot([rx0, rxF], [ry0, ryF], color=cMapTheme(0), linestyle='-.', linewidth=reflected_lw, label='Reflected', zorder=2)
  ax.plot([rx0 + rayAdj, rxF + rayAdj], [ry0, ryF], color=cMapTheme(0), linestyle='-.', linewidth=0.25, zorder=2)
  ax.plot([rx0 - rayAdj, rxF - rayAdj], [ry0, ryF], color=cMapTheme(0), linestyle='-.', linewidth=0.25, zorder=2)

  # refracted ray
  thetaPDeg = None # initialize emergent angle
  try:
    sin_angle_refraction = (n * math.sin(thetaRad)) / np
    if abs(sin_angle_refraction) <= 1: # Check for total internal reflection
      thetaPRad = math.asin(sin_angle_refraction)
      thetaPDeg = math.degrees(thetaPRad)

      ex0 = 0
      ey0 = 0

      exF = rayLength * math.sin(thetaPRad)
      eyF = -rayLength * math.cos(thetaPRad)

      ax.plot([ex0, exF], [ey0, eyF], color=cMapTheme(0), linestyle=':', linewidth=refracted_lw,
              label='Emergent', zorder=2)
      ax.plot([ex0 + rayAdj, exF + rayAdj], [ey0, eyF], color=cMapTheme(0), linestyle=':', linewidth=0.25, zorder=2)
      ax.plot([ex0 - rayAdj, exF - rayAdj], [ey0, eyF], color=cMapTheme(0), linestyle=':', linewidth=0.25, zorder=2)

    else:
      ax.text(0, -0.9, 'Total Internal Reflection.', color=cMapTheme(0), ha='center',
              zorder=2)

  except ValueError: # Handle cases where asin input is out of range (-1, 1)
    ax.text(0, -0.9, 'Total Internal Reflection.', color=cMapTheme(0), ha='center',
            zorder=2)

  ax.legend(loc='upper right', labelcolor=cMapTheme(7), facecolor= cMapTheme(5),
            edgecolor=cMapTheme(0))
  plt.show()

  # print statements for angles and reflection proportion
  print(f"Incident Angle: {theta:.2f} degrees")
  print(f"Reflected Angle: {theta:.2f} degrees")
  print(f"Proportion of Reflected Light: {R:.3f}")
  if thetaPDeg is not None:
    print(f"Emergent Refracted Angle: {thetaPDeg:.2f} degrees")
  else:
    print("Total Internal Reflection.")

def drawSphSection(r, chordLength, showConcave):

    if chordLength > 2 * r:
        print("Chord length (2y) cannot be greater than the sphere's diameter (2r).")
        return

    ySecMax = chordLength / 2
    xPlane = numpy.sqrt(r**2 - ySecMax**2)
    s = round(r - xPlane, 3) # compute sagittal depth
    print('The sagittal depth is', s, 'mm.')

    # create three dimensional plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # generate sphere data
    u = numpy.linspace(0, 2 * numpy.pi, 100) # azimuthal angle
    v = numpy.linspace(0, numpy.pi, 100)    # polar angle

    xCoords = r * numpy.outer(numpy.cos(u), numpy.sin(v))
    yCoords = r * numpy.outer(numpy.sin(u), numpy.sin(v))
    zCoords = r * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))

    xCapMask = xCoords >= xPlane
    xCap = numpy.where(xCapMask, xCoords, numpy.nan)
    yCap = numpy.where(xCapMask, yCoords, numpy.nan)
    zCap = numpy.where(xCapMask, zCoords, numpy.nan)

    # plot the spherical cap
    ax.plot_surface(xCap, yCap, zCap, color = cMapTheme(2), alpha = 0.60,
                    rstride = 1, cstride = 1, linewidth = 0.5, zorder = 0)

    # plot the cutting plane (circular base of the cap)
    theta = numpy.linspace(0, 2 * numpy.pi, 100)
    xBase = xPlane * numpy.ones_like(theta)
    yBase = ySecMax * numpy.cos(theta)
    zBase = ySecMax * numpy.sin(theta)
    # ax.plot(xBase, yBase, zBase, color = 'blue', linewidth = 2, linestyle = '--')

    # center of curvature
    ax.scatter(0, 0, 0, color =  cMapTheme(0), s = 50, label = 'C')

    # radius of curvature
    ax.plot([0, r], [0, 0], [0, 0], color= cMapTheme(3), linewidth=2, linestyle='-')
    ax.text(r/2, 0, 0, 'r', color = cMapTheme(3), fontsize = 12, ha = 'center') # Adjusted z-coordinate

    # sagittal depth
    ax.plot([xPlane, r], [0, 0], [0, 0], color = cMapTheme(4), linewidth = 2, linestyle = '-')
    ax.text((xPlane + r)/2, 0, 0, 's', color = cMapTheme(4), fontsize = 12, ha = 'center')

    # chord
    ax.plot([xPlane, xPlane], [ySecMax, -ySecMax], [0, 0], color = cMapTheme(5), linewidth = 2, linestyle = '-')
    ax.text(xPlane, 0, 0, '2y', color = cMapTheme(5), fontsize = 12, ha = 'center')

    # dynamic plot limits
    xMinLim = min(0, xPlane)
    xMaxLim = r
    yzMaxLim = ySecMax

    # add a padding based on the largest dimension, with a minimum value
    maxVal = max(xMaxLim - xMinLim, 2 * yzMaxLim)
    padding = 0.05 * maxVal

    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.grid(False)

    # set the viewpoint
    if showConcave == False:
      ax.view_init(elev = 10, azim = 130)
    else:
      ax.view_init(elev = 10, azim = -50)

    plt.show()

def drawWave(v, lam, f):
  if v == None:
    v = lam * f # m/s
    answer = v
  elif lam == None:
    lam = v / f # m
    answer = lam
  elif f == None:
    f = v / lam # Hz
    answer = f
  a = 1.0 # amplitude

  x0 = 0
  xF = 10 # meters
  numPoints = 500

  fig, ax = plt.subplots(figsize=(6, 2.5))
  ax.set_xlim(x0, xF)
  ax.set_ylim(-a * 1.5, a * 1.5)
  ax.set_xlabel("Position (m)")
  ax.set_ylabel("Amplitude")
  ax.grid(True)

  x = numpy.linspace(x0, xF, numPoints)
  line, = ax.plot(x, numpy.sin(2 * numpy.pi * x / lam), color = cMapTheme(0), lw=2)

  # add a single point to track a crest moving with the wave
  # Initial x-position of the tracked crest at t=0
  x_initial_crest = lam / 4 # Start at the first crest position
  y_crest = a # a crest's y-position is the amplitude

  point, = ax.plot([x_initial_crest], [y_crest], 'o', color= cMapTheme(2), markersize=8)

  def update(frame):
      t = frame * 0.05 # adjust this factor to control animation speed

      k = 2 * numpy.pi / lam # angular wave number
      omega = 2 * numpy.pi * f # angular frequency

      # sine wave equation: y(x, t) = A * sin(kx - omega*t)
      y = a * numpy.sin(k * x - omega * t)
      line.set_ydata(y)

      # Update the position of the tracked point (moving with the wave)
      x_particle_current = x_initial_crest + v * t
      point.set_data([x_particle_current], [y_crest]) # Modified here

      return (line, point)

  # create the animation
  # `frames` controls how many updates occurs
  # `interval` is the delay between frames in milliseconds
  ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
  plt.close()
  return round(answer, 3), ani


def drawWavefronts(n, p, radii, showRad, showVerg, showRays):
  #> inputs
  # p is an x, y tuple representing the object or image point
  # radii is a list of all radii

  # create a figure and a set of subplots
  fig, ax = plt.subplots(figsize = (6, 4))

  #> local variable definition
  numRadii = len(radii)
  maxRad = abs(max(radii, key = abs))

  #> draw light direction convention
  ax.quiver(xLim[0], yLim[0] * 0.95, xLim[1] - xLim[0], 0, color = cMapTheme(0), scale_units = 'xy',
            scale = 1, width = 0.003, zorder = 2)

  #> plot optical axis
  ax.plot([xLim[0], xLim[1]], [0, 0], color = cMapTheme(0), linewidth = 0.5,
           zorder = 0)

  #> draw point
  ax.scatter(p[0], p[1], c = 'k')

  #> draw wavefronts
  for iR in range(0, numRadii):
    r = radii[iR]
    V = computeVergence(n, r)
    # convergent wavefront
    if r > 0:
      theta = numpy.linspace(90, 270, 100) # angles
    # divergent wavefront
    else:
      theta = numpy.linspace(270, 90, 100) # angles
    w = (p[0] - r, 0) # location
    x = p[0] + r * numpy.cos(numpy.radians(theta)) # xCoords
    y = r * numpy.sin(numpy.radians(theta)) # yCoords
    ax.plot(x, y, color = cMapTheme(0))
    labelStr = ["$W_", str(letters[iR]), "$"]
    strCat = "".join(labelStr)
    ax.text(w[0], textY, strCat, color = cMapTheme(7), ha = 'center', bbox = tbox)

    #> draw radii
    if showRad == 1:
      ax.quiver(p[0] -r, p[1], r, 0, color = cMapTheme(4),
                 scale_units = 'xy', angles = 'xy', scale = 1, width = 0.005)

    if showVerg == 1:
      print("Wavefront ", str(letters[iR]), " has a vergence of", V, "D.")

    #> draw rays
    if showRays == 1:
      if r < 0:
        # divergent, start at point
        ax.quiver(p[0], p[1], x[50] - p[0], p[1] - y[0], color = cMapTheme(0),
                  scale_units = 'xy', angles = 'xy', scale = 1, width = 0.001)
        ax.quiver(p[0], p[1], x[50] - p[0], p[1] - y[99], color = cMapTheme(0),
                  scale_units = 'xy', angles = 'xy', scale = 1, width = 0.001)
      elif r > 0:
        # convergent, start at wavefronts
        ax.quiver(x[50], y[0], p[0] - x[50], p[1] - y[0], color = cMapTheme(0),
                  scale_units = 'xy', angles = 'xy', scale = 1, width = 0.001)
        ax.quiver(x[50], y[99], p[0] - x[50], p[1] - y[99], color = cMapTheme(0),
                  scale_units = 'xy', angles = 'xy', scale = 1, width = 0.001)

  # set plot details and display
  ax.set_aspect('equal')
  ax.set_xlim(xLim)
  ax.set_ylim(yLim)

  medColor = getInterpolatedColor(n)
  fig.set_facecolor(medColor)
  ax.axis('off')
  plt.show()

#> MAKES

def makeDualGraph(xList, yListA, yListB):

  fig, ax1 = plt.subplots(figsize = (6, 6))

  # left axis
  ax1.plot(xList[0], yListA[0], color = cMapTheme(3))
  ax1.set_ylabel(yListA[1], color= cMapTheme(3))
  ax1.set_xlabel(xList[1])

  if xList[0][0] > xList[0][-1]:
    ax1.invert_xaxis()

  # right axis
  ax2 = ax1.twinx()
  ax2.plot(xList[0], yListB[0], color = cMapTheme(5))
  ax2.set_ylabel(yListB[1], color = cMapTheme(5))
  ax2.set_xlabel(xList[1])

  ax1.set_frame_on(False)
  ax2.set_frame_on(False)
  plt.show()

def makeGraph(listA, listB, loc):
  
  fig, ax = plt.subplots(figsize = (6, 6))

  if listB == None:
    ax.plot(listA[0], listA[1], color = cMapTheme(3), lw = 2)
    ax.set_xlabel(listA[2])
    ax.set_ylabel(listA[3])
  else:
    ax.plot(listA[0], listA[1], color = cMapTheme(3), lw = 2, label = listA[4])

    ax.plot(listB[0], listB[1], color = cMapTheme(3), lw = 2, linestyle = 'dashed',
             label = listB[4])
    ax.set_xlabel(listB[2])
    ax.set_ylabel(listB[3]);

    ax.legend(loc = loc, labelcolor=cMapTheme(7), facecolor= cMapTheme(5),
            edgecolor=cMapTheme(0))

  ax.set_frame_on(False)
  plt.show()

def makeIncEm(n, l, lp): 

  #> INCIDENT WAVEFRONTS
  incLim = -1 # define leftmost location
  # define distances (from object) and radii for all incident wavefronts
  distListInc = numpy.linspace(incLim, 0, 101)
  if l < 0: # real object
    rListInc = numpy.linspace(l -incLim, l, 101)
  elif l > 0: # virtual object
    rListInc = numpy.linspace(l - incLim, l, 101)
  vListInc = n/rListInc
  L = round(vListInc[-1], 3)
  print('The vergence incident on the lens is:', L, 'D.')

  xList1 = [distListInc, "$W_I$ Distance from Lens (m)"]
  yListA1 = [rListInc, "$r_I$"]
  yListB1 = [vListInc, "$V_I$"]

  #> EMERGENT WAVEFRONTS
  emLim = 1 # define rightmost location
  # define distances (to image) and radii for all emergent wavefronts
  # real image and virtual images
  distListEm = numpy.linspace(0, emLim, 101)
  rListEm = numpy.linspace(lp, lp - emLim, 101)
  vListEm = n/rListEm
  Lp = round(vListEm[0], 3)
  print('The vergence emergent from the lens is:', Lp, 'D.')

  xList2 = [distListEm, "$W_E$ Distance from Lens (m)"]
  yListA2 = [rListEm, "$r_E$"]
  yListB2 = [vListEm, "$V_E$"]

  #> POWER
  F = round(Lp - L, 3)
  print('The power of the lens is:', F, 'D.')

  fig, axs = plt.subplots(1, 2, layout = 'constrained', figsize = (6, 4))

  # zero lines
  axs[0].plot(xList1[0], numpy.zeros(len(yListA1[0])),
            color = cMapTheme(0), linewidth = 0.5, zorder = 0)

  axs[1].plot(xList2[0], numpy.zeros(len(yListA2[0])),
            color = cMapTheme(0), linewidth = 0.5, zorder = 0)

  # left axis for first graph
  axs[0].plot(xList1[0], yListA1[0], color = cMapTheme(3))
  axs[0].set_ylabel(yListA1[1], color = cMapTheme(3))
  axs[0].set_xlabel(xList1[1])
  axs[0].set_xticks([xList1[0][0], xList1[0][50], xList1[0][-1]])
  axs[0].set_xticklabels([str(round(xList1[0][0], 3)),
                          str(round(xList1[0][50], 3)), 'Lens'], rotation=45)
  axs[0].set_frame_on(False)

  if (xList1[0][0] > xList1[0][-1]) and (l > 0):
    axs[0].invert_xaxis()

  # right axis for first graph
  ax0R = axs[0].twinx()
  ax0R.plot(xList1[0], yListB1[0], color = cMapTheme(5))
  ax0R.set_ylabel(yListB1[1], color = cMapTheme(5))
  ax0R.set_frame_on(False)

  # left axis for second graph
  axs[1].plot(xList2[0], yListA2[0], color = cMapTheme(3))
  axs[1].set_ylabel(yListA2[1], color = cMapTheme(3))
  axs[1].set_xlabel(xList2[1])
  axs[1].set_xticks([xList2[0][0], xList2[0][50], xList2[0][-1]])
  axs[1].set_xticklabels(['Lens', str(round(xList2[0][50], 3)),
                          str(round(xList2[0][-1], 3))], rotation=45)
  axs[1].set_frame_on(False)

  if xList2[0][0] > xList2[0][-1]:
    axs[1].invert_xaxis()

  # right axis for second graph
  ax1R = axs[1].twinx()
  ax1R.plot(xList2[0], yListB2[0], color = cMapTheme(5))
  ax1R.set_ylabel(yListB2[1], color = cMapTheme(5))
  ax1R.set_xlabel(xList2[1])
  ax1R.set_frame_on(False)

  axs[0].set_ylim((incLim, emLim));
  ax0R.set_ylim((-15, 15));
  axs[1].set_ylim((incLim, emLim));
  ax1R.set_ylim((-15, 15));

  # object text
  if l < 0: # real
    axs[0].text(l, 0, 'RO', color = cMapTheme(7), ha = 'center',
                bbox = tbox, zorder = 10)
  elif l > 0: # virtual
    axs[1].text(l, 0, 'VO', color = cMapTheme(7), ha = 'center',
                bbox = tbox, zorder = 10)

  # image text
  if lp > 0: # real
    axs[1].text(lp, 0, 'RI', color = cMapTheme(7), ha = 'center',
                bbox = tbox, zorder = 10)
  elif lp < 0: # virtual
    axs[0].text(lp, 0, 'VI', color = cMapTheme(7), ha = 'center',
                bbox = tbox, zorder = 10)
    
  #> UTILITIES

def devToPD(devDeg):
    devRad = numpy.radians(devDeg)
    dispM = 1 * math.tan(devRad) # assume distM is 1 meter
    dispCM = dispM * 100 # convert to centimeters
    pd = dispCM
    return pd

def getInterpolatedColor(refractiveIndex):
  # clamp refractiveIndex for color calculation between 1 and 3
  refractiveIndexClamped = max(1.0, min(3.0, refractiveIndex))

  if refractiveIndexClamped <= 2:
    t = (refractiveIndexClamped - 1.0) / (2.0 - 1.0) # t goes from 0 to 1 as n goes from 1 to 2
    return (
        cMapIndex(0)[0] * (1 - t) + cMapIndex(halfIndexColors)[0] * t,
        cMapIndex(0)[1] * (1 - t) + cMapIndex(halfIndexColors)[1] * t,
        cMapIndex(0)[2] * (1 - t) + cMapIndex(halfIndexColors)[2] * t
    )
  else: # refractiveIndexClamped > 2
    t = (refractiveIndexClamped - 2.0) / (3.0 - 2.0) # t goes from 0 to 1 as n goes from 2 to 3
    return (
        cMapIndex(halfIndexColors)[0] * (1 - t) + cMapIndex(numIndexColors-1)[0] * t,
        cMapIndex(halfIndexColors)[1] * (1 - t) + cMapIndex(numIndexColors-1)[1] * t,
        cMapIndex(halfIndexColors)[2] * (1 - t) + cMapIndex(numIndexColors-1)[2] * t)

def magReporter(L, F, Lp, LM):
  if L > 0:
    print('This virtual object and...')
  elif L < 0:
    print('This real object and...')

  if F > 0:
    print('this convergent lens create...')
  elif F < 0:
    print('this divergent lens create...')

  if Lp > 0:
    print('a real image that is...')
  elif Lp < 0:
    print('a virtual image that is...')

  if abs(LM) > 1:
    if LM > 0:
      print('erect and magnified.')
    elif LM < 0:
      print('inverted and magnified.')
  elif abs(LM) < 1:
    if LM > 0:
      print('erect and minified.')
    elif LM < 0:
      print('inverted and minified.')

def magReporterMerid(L, F, Lp, LM):
  if L > 0:
    print('This virtual object and...')
  elif L < 0:
    print('This real object and...')

  if F > 0:
    print('this convergent meridian create...')
  elif F < 0:
    print('this divergent meridian create...')

  if Lp > 0:
    print('a real image that is...')
  elif Lp < 0:
    print('a virtual image that is...')

  if abs(LM) > 1:
    if LM > 0:
      print('erect and magnified.')
    elif LM < 0:
      print('inverted and magnified.')
  elif abs(LM) < 1:
    if LM > 0:
      print('erect and minified.')
    elif LM < 0:
      print('inverted and minified.')

def pdToDev(pd):
  dispCM = pd # assume a distance of 1 m
  dispM = dispCM / 100 # convert to meters
  devDeg = numpy.degrees(math.atan(dispM))
  return devDeg

def returnGraph(listA, listB, loc):
  
  fig, ax = plt.subplots(figsize = (6, 6))

  if listB == None:
    ax.plot(listA[0], listA[1], color = cMapTheme(3), lw = 2)
    ax.set_xlabel(listA[2])
    ax.set_ylabel(listA[3])
  else:
    ax.plot(listA[0], listA[1], color = cMapTheme(3), lw = 2, label = listA[4])

    ax.plot(listB[0], listB[1], color = cMapTheme(3), lw = 2, linestyle = 'dashed',
             label = listB[4])
    ax.set_xlabel(listB[2])
    ax.set_ylabel(listB[3]);

    ax.legend(loc = loc, labelcolor=cMapTheme(7), facecolor= cMapTheme(5),
            edgecolor=cMapTheme(0))

  ax.set_frame_on(False)
  return ax

def safeDivide(num, den):
  if den == 0:
      return math.inf if num > 0 else -math.inf
  return num / den

def typeReporter(L, Lp, lensNum): 
  if L > 0 and Lp > 0:
    print('Virtual object and real image for lens', lensNum, '.')
  elif L > 0 and Lp < 0:
    print('Virtual object and virtual image for lens', lensNum, '.')
  elif L < 0 and Lp < 0:
    print('Real object and virtual image for lens', lensNum, '.')
  elif L < 0 and Lp > 0:
    print('Real object and real image for lens', lensNum, '.')
  elif L == 0 and Lp > 0:
    print('Parallel incidence and real image for lens', lensNum, '.')
  elif L == 0 and Lp < 0:
    print('Parallel incidence and virtual image for lens', lensNum, '.')
  elif L == 0 and Lp == 0:
    print('Parallel incidence and emergence for lens', lensNum, '.')
  elif L > 0 and Lp == 0:
    print('Virtual object and parallel emergence for lens', lensNum, '.')
  elif L < 0 and Lp == 0:
    print('Real object and parallel emergence for lens', lensNum, '.')

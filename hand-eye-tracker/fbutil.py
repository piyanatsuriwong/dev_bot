""" FrameBuffer utility class with additional drawing functions.

	Takes the FrameBuffer as parameter. call the methods and they will be applied to the FrameBuffer.

	triangle() are based on the adafruit micropython gfx library:
	See repository: https://github.com/adafruit/micropython-adafruit-gfx
	see repository: https://raw.githubusercontent.com/peter-l5/framebuf2
	see repository: https://github.com/adafruit/Adafruit-GFX-Library/blob/master/Adafruit_GFX.cpp
"""
__version__ = '0.1.0'

import math
try:
	import pygame
	PYGAME_AVAILABLE = True
except:
	PYGAME_AVAILABLE = False

class FBUtil:
	def __init__(self, fb ):
		self.fb = fb
		self._is_pygame = PYGAME_AVAILABLE and isinstance(fb, pygame.Surface)
	
	def _pixel(self, x, y, c):
		if self._is_pygame:
			self.fb.set_at((int(x), int(y)), c)
		else:
			self._pixel(x, y, c)
	
	def _hline(self, x, y, w, c):
		if self._is_pygame:
			pygame.draw.line(self.fb, c, (int(x), int(y)), (int(x+w), int(y)))
		else:
			self._hline(x, y, w, c)
	
	def _vline(self, x, y, h, c):
		if self._is_pygame:
			pygame.draw.line(self.fb, c, (int(x), int(y)), (int(x), int(y+h)))
		else:
			self._vline(x, y, h, c)
	
	def _rect(self, x, y, w, h, c):
		if self._is_pygame:
			pygame.draw.rect(self.fb, c, (int(x), int(y), int(w), int(h)), 1)
		else:
			self._rect(x, y, w, h, c)
	
	def _fill_rect(self, x, y, w, h, c):
		if self._is_pygame:
			pygame.draw.rect(self.fb, c, (int(x), int(y), int(w), int(h)), 0)
		else:
			self._fill_rect(x, y, w, h, c)

	def _peri_x(self, x, degrees, radius):
		sin = math.sin(math.radians(degrees))
		x = int(x+(radius*sin))
		return x

	def _peri_y(self, y, degrees, radius):
		cos = math.cos(math.radians(degrees))
		y = int(y-(radius*cos))
		return y

	def circle( self, x, y, radius, color, border=1, degrees=360, startangle=0):
		border = 5 if border > 5 else border
		# adding startangle to degrees
		if startangle > 0:
			degrees += startangle
		if border > 1:
			x = x - border//2
			y = y - border//2
			radius = radius-border//2
		for i in range(startangle, degrees):
			X = self._peri_x(x, i, radius)
			Y = self._peri_y(y, i, radius)
			if   i == 90:  X = X-1
			elif i == 180: Y = Y-1
			if border==1:
				self._pixel( X,Y, color )
			else:
				self._rect(X, Y, border, border, color )

	def fill_circle(self, x, y, radius, color):
		tempY = 0
		for i in range(180):
			xNeg = self._peri_x(x, 360-i, radius-1)
			xPos = self._peri_x(x, i, radius)
			if i > 89:
				Y = self._peri_y(y, i, radius-1)
			else:
				Y = self._peri_y(y, i, radius+1)
			if i == 90: xPos = xPos-1
			if tempY != Y and tempY > 0:
				length = xPos+1
				self._hline(xNeg, Y, length-xNeg, color ) # tick=4
			tempY = Y

	def fill_circle_helper( self, x0, y0, r, corners, delta, c ):
		# from https://github.com/adafruit/Adafruit-GFX-Library/blob/master/Adafruit_GFX.cpp
		f = 1 - r
		ddF_x = 1
		ddF_y = -2 * r
		x = 0
		y = r
		px = x
		py = y

		delta += 1 # Avoid some +1's in the loop

		while x < y:
			if f >= 0:
				y -= 1
				ddF_y += 2
				f += ddF_y

			x += 1
			ddF_x += 2
			f += ddF_x
			# These checks avoid double-drawing certain lines, important
			# for the SSD1306 library which has an INVERT drawing mode.
			if x < (y + 1):
				if (corners & 1) :
					self._vline(x0 + x, y0 - y, 2 * y + delta, c)					
				if (corners & 2)==2 :
					self._vline(x0 - x, y0 - y, 2 * y + delta, c)				

			if y != py :
				if (corners & 1) :
					self._vline(x0 + py, y0 - px, 2 * px + delta, c)
				if (corners & 2) :
					self._vline(x0 - py, y0 - px, 2 * px + delta, c)
				py = y

			px = x


	def oval( self, x, y, xradius, yradius, color, border=1, degrees=360, startangle=0):
		border = 5 if border > 5 else border
		# adding startangle to degrees
		if startangle > 0:
			degrees += startangle
		if border > 1:
			x = x - border//2
			y = y - border//2
			xradius = xradius-border//2
			yradius = yradius-border//2
		for i in range(startangle, degrees):
			X = self._peri_x(x, i, xradius)
			Y = self._peri_y(y, i, yradius)
			if   i == 90:  X = X-1
			elif i == 180: Y = Y-1
			if border==1:
				self._pixel( X,Y, color )
			else:
				self._rect(X, Y, border, border, color )

	def fill_oval( self, x, y, xradius, yradius, color ):
		tempY = 0
		for i in range(180):
			xNeg = self._peri_x(x, 360-i, xradius)
			xPos = self._peri_x(x, i, xradius)
			Y	= self._peri_y(y, i, yradius)

			if i > 89: Y = Y-1
			if tempY != Y and tempY > 0:
				length = xPos+1
				self._hline(xNeg, Y, length-xNeg, color ) # tick=4
			tempY = Y

	def rrect( self, x,y, width, height, radius, color ):
		max_r = (width if width<height else height)//2
		if max_r < radius:
			radius = max_r
		self.circle( x+radius, y+radius, radius, color, border=1, degrees=90, startangle=270)
		self._hline( x+radius, y, width-(2*radius), color )
		self.circle( x+width-radius, y+radius, radius, color, border=1, degrees=90, startangle=0)
		self._vline( x+width-1, y+radius, height-(2*radius), color )
		self.circle( x+width-radius, y+height-radius, radius, color, border=1, degrees=90, startangle=90)
		self._hline( x+radius+1, y+height, width-(2*radius), color )
		self.circle( x+radius, y+height-radius, radius, color, border=1, degrees=90, startangle=180)
		self._vline( x, y+radius+1, height-(2*radius)-1, color )

	def fill_rrect( self, x,y, width, height, radius, color ):
		max_r = (width if width < height else height) // 2
		if max_r < radius:
			radius = max_r
		self._fill_rect( x+radius, y, width-2*radius, height, color )
		self.fill_circle_helper( x+width-radius-1, y+radius, radius, 1, height-2*radius-1, color)
		self.fill_circle_helper( x+radius, y+radius, radius, 2, height-2*radius-1, color)

	def fill_triangle(self, x0, y0, x1, y1, x2, y2, c ):
		""" Triangle drawing function.  Will draw a single pixel wide triangle around the points (x0, y0), (x1, y1), and (x2, y2), colour c """
		if y0 > y1:
			y0, y1 = y1, y0
			x0, x1 = x1, x0
		if y1 > y2:
			y2, y1 = y1, y2
			x2, x1 = x1, x2
		if y0 > y1:
			y0, y1 = y1, y0
			x0, x1 = x1, x0
		a = 0
		b = 0
		last = 0
		if y0 == y2:
			a = x0
			b = x0
			if x1 < a:
				a = x1
			elif x1 > b:
				b = x1
			if x2 < a:
				a = x2
			elif x2 > b:
				b = x2
			self._hline(a, y0, b - a + 1, c)
			return
		dx01 = x1 - x0
		dy01 = y1 - y0
		dx02 = x2 - x0
		dy02 = y2 - y0
		dx12 = x2 - x1
		dy12 = y2 - y1
		if dy01 == 0:
			dy01 = 1
		if dy02 == 0:
			dy02 = 1
		if dy12 == 0:
			dy12 = 1
		sa = 0
		sb = 0
		y = y0
		if y0 == y1:
			last = y1 - 1
		else:
			last = y1
		while y <= last:
			a = x0 + sa // dy01
			b = x0 + sb // dy02
			sa += dx01
			sb += dx02
			if a > b:
				a, b = b, a
			self._hline(a, y, b - a + 1, c)
			y += 1
		sa = dx12 * (y - y1)
		sb = dx02 * (y - y0)
		while y <= y2:
			a = x1 + sa // dy12
			b = x0 + sb // dy02
			sa += dx12
			sb += dx02
			if a > b:
				a, b = b, a
			self._hline(a, y, b - a + 1, c)
			y += 1

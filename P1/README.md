#CSC320 Project 1: The Prokudin-Gorskii Colour Photo Collection (Worth: 8%)

[Sergey Prokudin-Gorskii](http://en.wikipedia.org/wiki/Sergey_Prokudin-Gorsky "Sergey Prokudin-Gorskii")  was a pioneer of colour photography. In the early 20-th century, he travelled across Russia and took thousands of striking colour pictures. His technique consisted of taking three black-and-white pictures of the scene simultaneously, with each picture taken through a blue, green, or red filter. In 1948, his glass-plate negatives were purchased by the Library of Congress and are available [on the web](http://www.loc.gov/pictures/collection/prok/ "Prokudin-Gorskii's glass-plate negatives").

In this project, you will explore techniques to automatically combine the __inverted__ blue, green, and red-filtered negatives into one colour photo.

##The Input

As input, you will take in an image that contains the three __inverted__ negatives (top to bottom: blue, green, red). An example of an image that you should work on is [here](images/00757v.jpg "An ideal image"). The images could be of different sizes. The images you should work with are are available [here](http://www.cs.toronto.edu/~guerzhoy/320/proj1/images.zip" A .zip file with images").


##Part 0

Briefly describe the image data that you are working with for the report. (This can be just one or two sentences.) Include examples of the images.

##Part 1

Write code to extract the three black-and-white __inverted__ negatives from the input image and to produce a colour image by combining the three colour channels.

The three __inverted__ negatives can be aligned by matching the three black-and-white __inverted__ negatives to each other to determine the optimal alignment. I suggest matching the green and the red __inverted__ negatives to the blue __inverted__ negative. The matching can be performed by varying the displacement of the negative that is being matched from -10 to 10 pixels along both the x and the y directions to find the best displacement. The matching metrics could be Normalized Cross-Correlation (NCC) and Sum of Squared Differences (SSD).

Obtain the colour image for several examples by matching the __inverted__ negatives using both NCC and SSD.￼In your report, indicate which seems to work better. Are there any artefacts in the output? What may explain the artefacts
For Part 1, you may assume that the input image will be of size similar to that of 00757v.jpg 

###Hints

- Some tuning needs to be done before the matching works well: for example, if you do not crop out the borders, the matching is less reliable.
- The input images are negatives. If the brightness ranges from 0 to 255, then for a negative n, the actual channel is 255-n. UPDATE: The images are pre-__inverted__. There is no need to invert them again.
- NEW: the images won't display correctly if the maximum intensity is above 1 and they type is float. To convert an image i to uint8, go i.astype(uint8)
-- To increase the efficiency of your code, avoid using loops! For example, you should always prefer 

	c = np.dot(u, v) 
over

	s =0
	for x, y in zip(u, v): s += x*y

- Suggestion for checking your code: take any colour photo, and try to align its three colour channels. The optimal displacement for channels of a single photo will very likely be (0, 0) (though that's not guaranteed...).

##Part 2##

The technique from Part 1 will only work for small images, and will take too long for larger images. This problem can be solved by rescaling the images with scipy.misc.imresize(), matching the small versions of the images to obtain a rough estimate of the match, and only then matching the large versions of the images. This procedure could be repeated several times. Implement matching so that it works for larger images as well. In the report, describe your results and the runtimes that you obtain.

##What to submit##

The project should be done using Python 2 and should run on CDF. Your report should be in PDF format. You should use LaTeX to generate the report, and submit the .tex file as well. A sample template will be posted soon.
###Important:###

- Readability counts! If your code isn't readable or your report doesn't make sense, they are not that useful. In addition, the TA can't read them. You will lose marks for those things.

- It is perfectly fine to discuss general ideas with other people, *if you acknowledge ideas in your report that are not your own*. However, you must not look at other people's code, or show your code to other people, and you must not look at other people's reports, or show your report to other people. All of those things are academic offenses.

##Acknowledgements##
The project was created by [Alexei Efros](http://www.eecs.berkeley.edu/~efros/ "Alexei Efros").

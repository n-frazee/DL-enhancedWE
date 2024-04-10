
set terminal qt size 800,600 font ',16'
set xlabel 'min samples'; set ylabel 'epsilon'
unset key
set cbrange [:20]; set cblabel 'num clusters'

set palette defined ( 0 '#D73027',\
    	    	      1 '#F46D43',\
		      2 '#FDAE61',\
		      3 '#FEE08B',\
		      4 '#D9EF8B',\
		      5 '#A6D96A',\
		      6 '#66BD63',\
		      7 '#1A9850' )

plot [4:201] [.03:2.02] 'outliers.dat' u 1:2:3 lc palette pt 5 ps 2.3
# dezero


# Testing commands

For single testing 
$ python3 -m unittest '....py'

For all testing
$ python3 -m unittest discover ...

For testing with coverage 
$ coverage run --omit */dist-packages/*,*site-packages/*,*tests* -m unittest tests/....py 
or
$ coverage run --omit */dist-packages/*,*site-packages/*,*tests* -m unittest discover tests

$ coverage report -m

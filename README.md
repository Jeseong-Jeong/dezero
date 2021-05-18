# dezero


# Testing commands

For single testing \n
$ python3 -m unittest '....py'

For all testing \n
$ python3 -m unittest discover ...

For testing with coverage \n
$ coverage run --omit */site-packages/* -m unittest tests/....py or \n
$ coverage run --omit */site-packages/* -m unittest discover tests
and \n
$ coverage report -m

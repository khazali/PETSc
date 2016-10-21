
###
##  Simple file for selecting targets by wildcard search on either 
##    the target names or by the values of the targets
##   make search='%ex2%'  gives runex2
##           Note that search uses a regex like filtering with % as wildcard
##   make argsearch='foo'  gives runex2 and runex3
#
RUNTESTS=runex1 runex2 runex3 
RUNTESTS_ARGS=$(foreach runtest, $(RUNTESTS), $(runtest)_ARGS)

ifdef search
  NEW_RUNTESTS = $(filter $(search),$(RUNTESTS))
  ACTUAL_RUN=$(NEW_RUNTESTS)
else
  ifdef argsearch
    NEW_RUNTESTS=$(foreach rarg, $(RUNTESTS_ARGS), $(if $(filter $(argsearch),$(value $(rarg))), $(rarg)))
    ACTUAL_RUN=$(NEW_RUNTESTS)
  else
    ACTUAL_RUN=$(RUNTESTS)
  endif
endif

ACTUAL_RUN += report_tests


all:
	-@echo ">"$(ACTUAL_RUN)

# Shell script depend on the source
run%.sh : % 

# Tests depend on shell scripts
% : %.sh
	-@echo $(@)
#	-@./$@.sh $(SHARGS)

report_tests:
	-@report_tests.sh

runex1_ARGS="foo"
runex2_ARGS="bar"
runex3_ARGS="foo bar"


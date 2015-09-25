import config.base
import os
import sys
import string

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    return

  def configureClosure(self):
    '''Determine if Apple ^close syntax is supported in C'''
    includes = '#include <stdio.h>\n'
    body = 'int (^closure)(int);'
    with self.maskLanguage('C'):
      if self.checkLink(includes, body):
        self.addDefine('HAVE_CLOSURE','1')
    return

  def configure(self):
    self.executeTest(self.configureClosure)
    return

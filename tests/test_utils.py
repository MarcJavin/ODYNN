"""
.. module::
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from unittest import TestCase
from odin import utils
import os
import numpy as np


class TestOptim(TestCase):

    def test_set_dir(self):
        utils.set_dir('')
        dir = 'unittest'
        pre = utils._current_dir
        utils.set_dir(dir)
        post = utils._current_dir
        self.assertTrue(os.path.exists(post))
        self.assertEqual(post, utils._current_dir)
        self.assertEqual(post[len(pre)-1:], 'unittest/')

        utils.set_dir('test2')
        post = utils._current_dir
        self.assertTrue(os.path.exists(post))
        self.assertEqual(post, utils._current_dir)
        self.assertEqual(post[len(pre) - 1:], 'test2/')

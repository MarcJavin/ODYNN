"""
.. module::
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from unittest import TestCase
from odynn import utils
import os
import numpy as np


class TestOptim(TestCase):

    def test_set_dir(self):
        dir = 'unittest'
        pre = utils.RES_DIR + '/'
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

    def test_clamp(self):
        a = utils.clamp(10)
        self.assertEqual(a, 10)
        a = utils.clamp(300)
        self.assertEqual(a, 255)
        a = utils.clamp(-2.)
        self.assertEqual(a, 0)

    def test_colorscale(self):
        a = utils.colorscale('#FFFFFF', 3)
        self.assertEqual(a, '#ffffff')
        a = utils.colorscale('#aaaaaa', 0.5)
        self.assertEqual(a, '#555555')

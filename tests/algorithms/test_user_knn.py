import unittest

import ConfigSpace as CS

from lkauto.algorithms.user_knn import UserUser


class TestUserKNN(unittest.TestCase):
    def test_init_givenObjectInitialized_ObjectInitializedCorrectlyExpected(self):
        max_n = 100
        min_n = 5
        min_s = 0.001
        user_user = UserUser(max_n, min_nbrs=min_n, min_sim=min_s)
        self.assertIsInstance(user_user, UserUser)
        self.assertEqual(max_n, user_user.max_nbrs)
        self.assertEqual(min_n, user_user.min_nbrs)
        self.assertEqual(min_s, user_user.min_sim)

    def test_defaultConfigspace_GivenFunctionCalled_correctConfigSpaceReturnedExpected(self):
        cs = UserUser.get_default_configspace()
        params = list(cs.values())
        self.assertIsInstance(cs, CS.ConfigurationSpace)
        self.assertTrue(any(
            param.name == "max_nbrs" and param.default_value == 1000 and param.lower == 1 and param.upper == 10000 for
            param in params))
        self.assertTrue(any(
            param.name == "min_nbrs" and param.default_value == 1 and param.lower == 1 and param.upper == 1000 for
            param in params))
        self.assertTrue(any(
            param.name == "min_sim" and param.default_value == 1.0e-6 and param.lower == 1.0e-10 and param.upper ==
            1.0e-2 for param in params))


if __name__ == '__main__':
    unittest.main()

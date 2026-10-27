"""CPU analytic cases, not learned-model accuracy or physics experiments."""
import unittest

import torch

from constraint_transport import (
    carry_reference, filter_attachment, fit_rigid, fuse_constraints, rigid_references,
)


class TransportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.dtype = torch.float64
        self.free = torch.tensor([[[.1, -.2, .3]]], dtype=self.dtype)
        self.eye = torch.eye(3, dtype=self.dtype)

    def fuse(self, reference, strength, factors=None, free=None):
        a = len(strength)
        reference = torch.tensor(reference, dtype=self.dtype).reshape(1, 1, a, 3)
        if factors is None:
            factors = self.eye.expand(1, 1, a, 3, 3)
        return fuse_constraints(
            self.free if free is None else free, reference, factors,
            torch.tensor(strength, dtype=self.dtype).reshape(1, 1, a))

    def test_release_returns_free_including_falling_motion(self):
        value, conflict = self.fuse([[4, 5, 6], [8, 9, 10]], [0, 0])
        torch.testing.assert_close(value, self.free, rtol=0, atol=0)
        self.assertEqual(float(conflict), 0)

    def test_grasp_tracks_translation_and_rotation_about_world_origin(self):
        # A point offset from the origin must follow rotation, not just TCP shift.
        rotation = torch.tensor([[[[0., -1, 0], [1, 0, 0], [0, 0, 1]]]], dtype=self.dtype)
        translation = torch.tensor([[[.2, .3, .4]]], dtype=self.dtype)
        reference = rigid_references(self.free, rotation, translation)
        torch.testing.assert_close(reference[0, 0, 0],
                                   torch.tensor([.4, .4, .7], dtype=self.dtype))
        output, _ = fuse_constraints(self.free, reference, self.eye.reshape(
            1, 1, 1, 3, 3), torch.full((1, 1, 1), 1e4, dtype=self.dtype))
        self.assertLess(float((output - reference[:, :, 0]).norm()), 1e-4)

    def test_normal_constraint_preserves_tangential_free_motion(self):
        factors = torch.tensor([[[[[0.], [0.], [1.]]]]], dtype=self.dtype)
        value, _ = self.fuse([[5, 6, .8]], [100], factors)
        torch.testing.assert_close(value[..., :2], self.free[..., :2])
        self.assertGreater(float(value[..., 2]), .79)

    def test_two_identical_arms_do_not_double_object_motion(self):
        target = [[.2, .1, .6], [.2, .1, .6]]
        value, conflict = self.fuse(target, [100, 100])
        expected = (self.free + 200 * torch.tensor(target[:1], dtype=self.dtype)) / 201
        torch.testing.assert_close(value, expected.reshape_as(value))
        self.assertLess(float(conflict), .01)

    def test_incompatible_arms_report_disagreement_not_fake_success(self):
        value, conflict = self.fuse([[1, 0, 0], [-1, 0, 0]], [100, 100])
        self.assertGreater(float(conflict), 190)
        self.assertLess(abs(float(value[..., 0])), .001)

    def test_arm_rename_and_world_rigid_transform_equivariance(self):
        refs = torch.tensor([[[[1., 2, 3], [-2, 0, 1]]]], dtype=self.dtype)
        factors = torch.tensor([[[[[1., .2], [.3, .1], [0, .5]],
                                  [[.2, .1], [1, .4], [.3, .1]]]]], dtype=self.dtype)
        strength = torch.tensor([[[.8, .4]]], dtype=self.dtype)
        value, conflict = fuse_constraints(self.free, refs, factors, strength)
        swapped, other = fuse_constraints(self.free, refs.flip(2), factors.flip(2),
                                          strength.flip(2))
        torch.testing.assert_close(value, swapped)
        torch.testing.assert_close(conflict, other)
        q = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]], dtype=self.dtype)
        t = torch.tensor([.4, -.6, .9], dtype=self.dtype)
        moved, changed = fuse_constraints(
            self.free @ q.T + t, refs @ q.T + t, q @ factors, strength)
        torch.testing.assert_close(moved, value @ q.T + t)
        torch.testing.assert_close(conflict, changed)

    def test_persistence_and_explicit_release(self):
        prev = torch.tensor([.9, .1], dtype=self.dtype)
        zero = torch.zeros_like(prev)
        torch.testing.assert_close(filter_attachment(prev, zero, zero), prev)
        torch.testing.assert_close(filter_attachment(prev, zero, torch.ones_like(prev)), zero)
        torch.testing.assert_close(filter_attachment(zero, torch.ones_like(prev), zero),
                                   torch.ones_like(prev))

    def test_reanchoring_leaks_free_drift_but_retained_reference_does_not(self):
        # Constructed 7 mm bias, NOT an observed model-error measurement.
        free = torch.tensor([[[.007, 0, 0]]], dtype=self.dtype)
        anchored = torch.zeros_like(free)
        reanchored = torch.zeros_like(free)
        carrier = torch.zeros((1, 1, 1, 3), dtype=self.dtype)
        rotation = self.eye.reshape(1, 1, 3, 3)
        translation = torch.zeros((1, 1, 3), dtype=self.dtype)
        factors = self.eye.reshape(1, 1, 1, 3, 3)
        strength = torch.full((1, 1, 1), 100., dtype=self.dtype)
        for _ in range(10):
            carrier = carry_reference(carrier, anchored, torch.zeros_like(strength),
                                      rotation, translation)
            anchored, _ = fuse_constraints(free, carrier, factors, strength)
            ref = rigid_references(reanchored, rotation, translation)
            reanchored, _ = fuse_constraints(free, ref, factors, strength)
        torch.testing.assert_close(anchored[..., 0], torch.tensor([[.007 / 101]],
                                                                dtype=self.dtype))
        self.assertGreater(float(reanchored[..., 0]), 9 * float(anchored[..., 0]))
        renewed = carry_reference(carrier, free, torch.ones_like(strength),
                                  rotation, translation)
        torch.testing.assert_close(renewed[:, :, 0], free)

    def test_operator_gradients_reach_reference_strength_and_direction(self):
        free = self.free.clone().requires_grad_()
        refs = torch.tensor([[[[.2, .5, -.3]]]], dtype=self.dtype, requires_grad=True)
        factors = torch.tensor([[[[[1., .2], [.3, .4], [.5, .1]]]]],
                               dtype=self.dtype, requires_grad=True)
        strength = torch.tensor([[[.7]]], dtype=self.dtype, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda f, r, l, s: fuse_constraints(f, r, l, s)[0],
            (free, refs, factors, strength)))

    def test_fit_uses_correspondence_and_rejects_collinear_points(self):
        x = torch.tensor([[[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                         dtype=self.dtype)
        q = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]], dtype=self.dtype)
        t = torch.tensor([.2, .3, .4], dtype=self.dtype)
        fit = fit_rigid(x, x @ q.T + t, torch.ones((1, 4), dtype=torch.bool))
        self.assertTrue(bool(fit.valid.all()))
        torch.testing.assert_close(fit.rotation[0], q)
        torch.testing.assert_close(fit.translation[0], t)
        line = torch.tensor([[[0., 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=self.dtype)
        failed = fit_rigid(line, line + t, torch.ones((1, 3), dtype=torch.bool))
        self.assertFalse(bool(failed.valid.any()))

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            self.fuse([[0, 0, 0]], [-1])
        with self.assertRaises(ValueError):
            filter_attachment(torch.tensor([1.1]), torch.zeros(1), torch.zeros(1))


if __name__ == "__main__":
    unittest.main()

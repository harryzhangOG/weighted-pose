import torch.nn as nn

mse_criterion = nn.MSELoss(reduction="sum")
huber_criterion = nn.HuberLoss()


class WeightedHybridLoss(nn.Module):
    def forward(self, action_pos, T_pred, T_gt, Fx, goal_flow):

        pred_flow_action = (T_gt.transform_points(action_pos) - action_pos).detach()
        goal_flow = goal_flow.reshape(-1, 2000, 3)[:, :500, :]

        loss = weighted_hybrid_loss(
            pred_T_action=T_pred,
            gt_T_action=T_gt,
            points_trans_action=action_pos,
            pred_flow_action=Fx,
            points_action=T_gt.transform_points(action_pos),
            goal_flow=goal_flow,
        )
        return loss


def weighted_hybrid_loss(
    pred_T_action,
    gt_T_action,
    points_trans_action,
    pred_flow_action,
    points_action,
    goal_flow,
    action_weight=1.0,
    smoothness_weight=0.1,
    consistency_weight=1.0,
):
    induced_flow_action = (
        pred_T_action.transform_points(points_trans_action) - points_trans_action
    ).detach()
    pred_points_action = pred_T_action.transform_points(
        points_trans_action
    )  # pred_points_action =  T0^-1@points_trans_action

    point_loss_action = mse_criterion(
        pred_points_action, points_action
    ) + mse_criterion(pred_T_action.get_matrix(), gt_T_action.get_matrix())

    point_loss = action_weight * point_loss_action

    dense_loss = dense_flow_loss(
        points=points_trans_action, flow_pred=pred_flow_action, trans_gt=gt_T_action
    ) + dense_flow_loss(
        points=points_trans_action, flow_pred=goal_flow, trans_gt=gt_T_action
    )

    # Loss associated flow vectors matching a consistent rigid transform
    smoothness_loss_action = mse_criterion(
        pred_flow_action,
        induced_flow_action,
    ) + mse_criterion(
        goal_flow,
        induced_flow_action,
    )

    smoothness_loss = action_weight * smoothness_loss_action

    loss = (
        point_loss
        # + smoothness_weight * smoothness_loss
        # + consistency_weight * dense_loss
    )

    return loss


def weighted_hybrid_loss_v2(
    pred_T_action,
    gt_T_action,
    points_trans_action,
    pred_flow_action,
    points_action,
    goal_flow,
    action_weight=1.0,
    smoothness_weight=0.1,
    consistency_weight=1.0,
):
    induced_flow_action = (
        pred_T_action.transform_points(points_trans_action) - points_trans_action
    ).detach()
    pred_points_action = pred_T_action.transform_points(
        points_trans_action
    )  # pred_points_action =  T0^-1@points_trans_action

    point_loss_action = mse_criterion(pred_points_action, points_action)

    point_loss = action_weight * point_loss_action

    dense_loss = dense_flow_loss(
        points=points_trans_action, flow_pred=pred_flow_action, trans_gt=gt_T_action
    ) + dense_flow_loss(
        points=points_trans_action, flow_pred=goal_flow, trans_gt=gt_T_action
    )

    # Loss associated flow vectors matching a consistent rigid transform
    smoothness_loss_action = mse_criterion(
        pred_flow_action,
        induced_flow_action,
    )

    smoothness_loss = action_weight * smoothness_loss_action

    loss = (
        point_loss
        + smoothness_weight * smoothness_loss
        + consistency_weight * dense_loss
    )

    return loss


class TAXPoseLoss(nn.Module):
    def forward(self, action_pos, T_pred, T_gt, Fx):

        pred_flow_action = (T_gt.transform_points(action_pos) - action_pos).detach()

        loss = taxpose_loss(
            pred_T_action=T_pred,
            gt_T_action=T_gt,
            points_trans_action=action_pos,
            pred_flow_action=Fx,
            points_action=T_gt.transform_points(action_pos),
        )
        return loss


def dense_flow_loss(points, flow_pred, trans_gt):
    flow_gt = trans_gt.transform_points(points) - points
    loss = mse_criterion(
        flow_pred,
        flow_gt,
    )
    return loss


def taxpose_loss(
    pred_T_action,
    gt_T_action,
    points_trans_action,
    pred_flow_action,
    points_action,
    action_weight=1.0,
    smoothness_weight=0.1,
    consistency_weight=1.0,
):
    induced_flow_action = (
        pred_T_action.transform_points(points_trans_action) - points_trans_action
    ).detach()
    pred_points_action = pred_T_action.transform_points(
        points_trans_action
    )  # pred_points_action =  T0^-1@points_trans_action

    # pred_T_action=T0^-1
    # gt_T_action = T0.inverse()

    point_loss_action = mse_criterion(pred_points_action, points_action)

    point_loss = action_weight * point_loss_action

    dense_loss = dense_flow_loss(
        points=points_trans_action, flow_pred=pred_flow_action, trans_gt=gt_T_action
    )

    # Loss associated flow vectors matching a consistent rigid transform
    smoothness_loss_action = mse_criterion(
        pred_flow_action,
        induced_flow_action,
    )

    smoothness_loss = action_weight * smoothness_loss_action

    loss = (
        point_loss
        + smoothness_weight * smoothness_loss
        + consistency_weight * dense_loss
    )

    return loss

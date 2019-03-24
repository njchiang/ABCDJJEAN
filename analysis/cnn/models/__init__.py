import tensorflow as tf


def create_keras_model(model, input_shape, dtype=tf.float32, weight_decay=1e-4):
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=dtype)

    logits = model(inputs)
    return tf.keras.models.Model(inputs=inputs, outputs=logits)

def create_estimator_from_subclass(model, cfg):
    dtype = tf.float32
    params = {}
    def model_fn(features, labels, mode, params, config=None):

        model_inst = model(**params)
        logits = model_inst(features, training=mode == tf.estimator.ModeKeys.TRAIN)
        tf.logging.info("Regularizing {} layers".format(len(model_inst.losses)))
        vols = features["image"]
        with tf.name_scope("inputs"):
            s = tf.shape(vols)
            dims = {1: "x", 2: "y", 3: "z"}
            im_list = []
            for d, n in dims.items():
                disp_image = tf.gather(
                    vols, indices=s[d] // 2, axis=d)
                im_list.append(tf.summary.image(
                    name="{}_view".format(n), tensor=disp_image,
                    max_outputs=1))
            tf.summary.merge(im_list)  # this should plot gradients?

        """ Model outputs """
        predictions = probabilities = logits

        if mode == tf.estimator.ModeKeys.PREDICT:
            prediction_output = {"Predicted-residual": predictions}
            return tf.estimator.EstimatorSpec(mode,
                                              predictions=prediction_output)

        """ Model metrics """
        losses = {}
        with tf.variable_scope("losses"):
            losses["mse"] = tf.losses.mean_squared_error(labels, logits)
            losses["reg"] = tf.add_n(model_inst.losses) if len(model_inst.losses) > 0 else 0.
            losses["total"] = loss =  tf.add(losses["mse"], losses["reg"])
            [tf.summary.scalar(h, v) for h, v in losses.items()]

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss)  # ,
                # evaluation_hooks=eval_hook_list)

        assert mode == tf.estimator.ModeKeys.TRAIN

        """ Training operations """
        var_list = tf.trainable_variables()
        learning_rate = tf.constant(cfg["learning_rate"])
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1)

        with tf.name_scope("training"):
            tf.summary.scalar("learning_rate", learning_rate)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        with tf.control_dependencies(model_inst.updates):
            grads_and_vars = \
                optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=tf.train.get_global_step())

        with tf.name_scope("gradient_hists"):
            grad_hist_list = [
                tf.summary.histogram(
                    v.name.replace(
                        ":", "_") + "/gradient", g
                ) for g, v in grads_and_vars]

            tf.summary.merge(grad_hist_list)  # this should plot gradients?

        with tf.name_scope("parameter_hists"):
            histogram_list = [
                tf.summary.histogram(
                    var.name.replace(":", "_"), var)
                for var in tf.trainable_variables()]

            tf.summary.merge(histogram_list)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return model_fn, params


def create_estimator_from_functional(model, cfg):
    dtype = tf.float32
    params = None
    def model_fn(features, labels, mode, params, config=None):

        logits, end_points, l2_loss = model(features, is_training=mode == tf.estimator.ModeKeys.TRAIN, **params)

        vols = features["image"]
        with tf.name_scope("inputs"):
            s = tf.shape(vols)
            dims = {1: "x", 2: "y", 3: "z"}
            im_list = []
            for d, n in dims.items():
                disp_image = tf.gather(
                    vols, indices=s[d] // 2, axis=d)
                im_list.append(tf.summary.image(
                    name="{}_view".format(n), tensor=disp_image,
                    max_outputs=1))
            tf.summary.merge(im_list)  # this should plot gradients?

        """ Model outputs """
        predictions = probabilities = logits

        if mode == tf.estimator.ModeKeys.PREDICT:
            prediction_output = {"Predicted-residual": k}
            return tf.estimator.EstimatorSpec(mode,
                                              predictions=prediction_output)

        """ Model metrics """
        losses = {}
        with tf.variable_scope("losses"):
            losses["mse"] = tf.losses.mean_squared_error(labels, logits)
            losses["reg"] = l2_loss
            losses["total"] = loss =  losses["mse"] + losses["reg"]
            [tf.summary.scalar(h, v) for h, v in losses.items()]

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss)  # ,
                # evaluation_hooks=eval_hook_list)

        assert mode == tf.estimator.ModeKeys.TRAIN

        """ Training operations """
        var_list = tf.trainable_variables()
        learning_rate = tf.constant(cfg["learning_rate"])
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1)

        with tf.name_scope("training"):
            tf.summary.scalar("learning_rate", learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads_and_vars = \
                optimizer.compute_gradients(loss, var_list=var_list)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=tf.train.get_global_step())

        with tf.name_scope("gradients"):
            grad_hist_list = [
                tf.summary.histogram(
                    v.name.replace(
                        ":", "_") + "/gradient", g
                ) for g, v in grads_and_vars]

            tf.summary.merge(grad_hist_list)  # this should plot gradients?

        with tf.name_scope("parameters"):
            histogram_list = [
                tf.summary.histogram(
                    var.name.replace(":", "_"), var)
                for var in tf.trainable_variables()]

            tf.summary.merge(histogram_list)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return model_fn, params

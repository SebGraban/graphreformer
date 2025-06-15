from graphgen.dfscode.dfs_wrapper import graph_from_dfscode
import tensorflow as tf 
import networkx as nx

import numpy as np

from keras.models import Model
from keras.metrics import Mean

class GraphAR(Model):

    def __init__(
        self,
        autoregessive_model,
        timestep_1,
        timestep_2,
        vertex_1,
        edge,
        vertex_2,
        features,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.autoregressive_model = autoregessive_model
        self.timestep_1 = timestep_1
        self.timestep_2 = timestep_2
        self.vertex_1 = vertex_1
        self.edge = edge
        self.vertex_2 = vertex_2

        self.loss = Mean(name='bce_loss')
    
        self.features = features

    def train_step(self, data):
        x, x_target, seq_lens = data
        batch_size = tf.shape(x)[0]
        max_seq_len = tf.shape(x)[1]

        initial_states = self.autoregressive_model.init_states(batch_size)

        with tf.GradientTape() as tape:
            mask = tf.sequence_mask(seq_lens, max_seq_len)

            # Input to the first LSTM layer
            current_input = x

            current_input, *initial_states = self.autoregressive_model(
                [current_input] + initial_states,
                training=True
            )

            # Final LSTM output from top layer
            rnn_output = current_input

            # Apply prediction heads
            timestamp1_logits = self.timestep_1(rnn_output)
            timestamp2_logits = self.timestep_2(rnn_output)
            vertex1_logits = self.vertex_1(rnn_output)
            edge_logits = self.edge(rnn_output)
            vertex2_logits = self.vertex_2(rnn_output)

            # Losses
            loss_t1 = self.component_loss(timestamp1_logits, 0, self.features['max_num_nodes'] + 1, mask, x_target)
            loss_t2 = self.component_loss(timestamp2_logits, self.features['max_num_nodes'] + 1, self.features['max_num_nodes'] + 1, mask, x_target)
            loss_v1 = self.component_loss(vertex1_logits, 2 * self.features['max_num_nodes'] + 2, len(self.features['node_vocab']) + 1, mask, x_target)
            loss_e = self.component_loss(edge_logits, 2 * self.features['max_num_nodes'] + 2 + len(self.features['node_vocab']) + 1, len(self.features['edge_vocab']) + 1, mask, x_target)
            loss_v2 = self.component_loss(vertex2_logits, 2 * self.features['max_num_nodes'] + 2 + len(self.features['node_vocab']) + 1 + len(self.features['edge_vocab']) + 1, len(self.features['node_vocab']) + 1, mask, x_target)

            total_loss = loss_t1 + loss_t2 + loss_v1 + loss_e + loss_v2
            total_non_pad = tf.reduce_sum(tf.cast(mask, tf.float32))
            total_loss = total_loss / total_non_pad

        # Backpropagation
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": total_loss}

    
    # Compute losses for each component
    def component_loss(self, logits, offset, num_classes, mask, targets):
        # Extract component values from the multi-hot encoded sequence
        component_targets = tf.argmax(
            targets[:, :, offset:offset+num_classes], 
            axis=-1
        )
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            component_targets, logits, from_logits=True
        )
        return tf.reduce_sum(tf.where(mask, loss, 0))
    

    def predict_graph(self, max_num_edges=100):
        max_nodes = self.features['max_num_nodes']
        len_node_vec = len(self.features['node_vocab']) + 1
        len_edge_vec = len(self.features['edge_vocab']) + 1
        feature_len = self.features['feature_len']

        batch_size = 1
        
        initial_states = self.autoregressive_model.init_states(batch_size)
        rnn_input = tf.zeros((batch_size, 1, feature_len), dtype=tf.float32)

        dfscode = []
        node_inv_map = {v: k for k, v in self.features['node_to_index'].items()}
        edge_inv_map = {v: k for k, v in self.features['edge_to_index'].items()}

        for i in range(max_num_edges):
            inputs = [rnn_input] + initial_states
            rnn_output, *new_states = self.autoregressive_model(inputs, training=False)
            initial_states = new_states  # Update states

            t1_logits = self.timestep_1(rnn_output)[:, 0, :]
            t2_logits = self.timestep_2(rnn_output)[:, 0, :]
            v1_logits = self.vertex_1(rnn_output)[:, 0, :]
            e_logits = self.edge(rnn_output)[:, 0, :]
            v2_logits = self.vertex_2(rnn_output)[:, 0, :]

            # Categorical sampling
            t1 = tf.random.categorical(t1_logits, 1)[0, 0].numpy()
            t2 = tf.random.categorical(t2_logits, 1)[0, 0].numpy()
            v1 = tf.random.categorical(v1_logits, 1)[0, 0].numpy()
            e = tf.random.categorical(e_logits, 1)[0, 0].numpy()
            v2 = tf.random.categorical(v2_logits, 1)[0, 0].numpy()

            # Stop conditions
            if t1 == max_nodes or t2 == max_nodes or \
               v1 == len_node_vec or v2 == len_node_vec:
                break

            # Decode
            dfscode.append((t1, t2, node_inv_map[v1], edge_inv_map[e], node_inv_map[v2]))

            # Build next input
            input_vec = np.zeros((1, 1, feature_len), dtype=np.float32)
            input_vec[0, 0, t1] = 1
            input_vec[0, 0, max_nodes + 1 + t2] = 1
            input_vec[0, 0, 2 * max_nodes + 2 + v1] = 1
            input_vec[0, 0, 2 * max_nodes + 2 + len_node_vec + e] = 1
            input_vec[0, 0, 2 * max_nodes + 2 + len_node_vec + len_edge_vec + v2] = 1

            rnn_input = tf.convert_to_tensor(input_vec)

        # Convert to NetworkX graph
        graph = graph_from_dfscode(dfscode)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        if len(graph.nodes):
            max_comp = max(nx.connected_components(graph), key=len)
            graph = nx.Graph(graph.subgraph(max_comp))

        return graph

    
    @property
    def metrics(self):
        return [self.loss]

class StackedLSTM(Model):
    def __init__(self, num_layers, lstm_units, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.lstm_layers = []
        for i in range(num_layers):
            lstm_layer = tf.keras.layers.LSTM(
                units=lstm_units,
                return_sequences=True,
                return_state=True,
                name=f'lstm_{i+1}'
            )
            self.lstm_layers.append(lstm_layer)


    def call(self, inputs, training=False):
        x = inputs[0]
        states = inputs[1:] if len(inputs) > 1 else None

        new_states = []
        for i in range(self.num_layers):
            if states is not None:
                h, c = states[2*i], states[2*i + 1]
            else:
                h = tf.zeros((tf.shape(x)[0], self.lstm_layers[i].units), dtype=tf.float32)
                c = tf.zeros((tf.shape(x)[0], self.lstm_layers[i].units), dtype=tf.float32)

            x, h, c = self.lstm_layers[i](x, initial_state=[h, c], training=training)
            new_states.extend([h, c])
        
        return x, *new_states

    def init_states(self, batch_size):
        hidden_states = []
        for i in range(self.num_layers):
            h = tf.zeros((batch_size, self.lstm_layers[i].units), dtype=tf.float32)
            c = tf.zeros((batch_size, self.lstm_layers[i].units), dtype=tf.float32)
            hidden_states.extend([h, c])
        return hidden_states

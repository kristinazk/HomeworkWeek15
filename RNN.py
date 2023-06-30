import numpy as np

class RNN:
    def __init__(self, input_size=27, hs_size=150, lr=0.001):
        self.alphabet = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                                  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '/'])
        self.input_size = input_size
        self.lr = lr
        self.hs_size = hs_size

        self.outputs = []

        # Bias initialization
        self.bias = np.zeros((hs_size, 1))
        self.bias_output = np.zeros((input_size, 1))

        # Initialization with Xavier-Glorot
        self.input_weight = np.random.normal(0, np.sqrt(2 / (hs_size + input_size), (hs_size, input_size)))
        self.hidden_weight = np.random.normal(0, np.sqrt(2 / (hs_size + hs_size), (hs_size, hs_size)))
        self.output_weight = np.random.normal(0, np.sqrt(2 / (input_size + hs_size), (input_size, hs_size)))

    @staticmethod
    def activation(X, activation='tangent'):
        if activation == 'tangent':
            return np.tanh(X)

        elif activation == 'softmax':
            e_x = np.exp(X - np.max(X))
            return e_x / e_x.sum()

    def forward_calc_rec(self, encoded_letter, hidden_state, index=0):
        z = self.bias + self.hidden_weight @ hidden_state + self.input_weight @ encoded_letter[index]

        output = self.activation(self.bias_output + self.output_weight @ z, 'softmax')

        if encoded_letter[index][-1] == 1:  # Checking for end of sequence symbol
            return output

        hidden_layer = self.activation(z)

        self.forward_calc_rec(encoded_letter, hidden_layer, index + 1)

        return output, hidden_layer

    def forward(self, X, state):
        self.seq_length = len(X)

        encoded_X = []
        outputs_y = []
        hidden_state = []

        for i in range(self.seq_length):
            for letter in X[i]:
                encoded = self.ohe(letter)
                encoded_X.append(encoded)
                y, state = self.forward_calc_rec(encoded, state)
                outputs_y.append(y)
                hidden_state.append(state)

        return encoded_X, outputs_y, hidden_state

    def backward(self, hidden_state, predicted_y, X, actual_y):
        input_w_deriv = np.zeros_like(self.input_weight)
        hidden_w_deriv = np.zeros_like(self.hidden_weight)
        output_w_deriv = np.zeros_like(self.output_weight)
        bias_deriv = np.zeros_like(self.bias)
        bias_output_deriv = np.zeros_like(self.bias_output)

        hidden_layer_next = np.zeros_like(hidden_state[0])

        for t in reversed(range(self.seq_length)):
            dy = np.copy(actual_y[t])

            dy[actual_y[t]] -= 1

            bias_output_deriv += dy[t]
            output_w_deriv += np.dot(dy[t], hidden_state[t].T)

            hidden_layer_deriv = np.dot(output_w_deriv.T, dy) + hidden_layer_next

            tanh_deriv = 1 - hidden_state[t] ** 2

            bias_deriv += tanh_deriv * hidden_layer_deriv

            hidden_w_deriv += np.dot(tanh_deriv * hidden_layer_deriv, hidden_state[t - 1].T)

            input_w_deriv += np.dot(tanh_deriv * hidden_layer_deriv, X[t].T)

            hidden_layer_next = np.dot(hidden_w_deriv.T, tanh_deriv * hidden_layer_deriv)

        return bias_output_deriv, bias_deriv, hidden_w_deriv, input_w_deriv, output_w_deriv

    def weights_update(self, db, dbo, dhw, diw, dow):
        self.input_weight -= self.lr * diw
        self.hidden_weight -= self.lr * dhw
        self.output_weight -= self.lr * dow
        self.bias -= self.lr * db
        self.bias_output -= self.lr * dbo

    def train(self, X, y_train, num_iter=100):
        for _ in range(num_iter):
            hidden_init = np.zeros((self.hs_size, 1))

            enc_X, outputs, hidden_state = self.forward(X, hidden_init)

            db, dbo, dhw, diw, dow = self.backward(hidden_state, outputs, enc_X, y_train)

            self.weights_update(db, dbo, dhw, diw, dow)

    def predict(self, X_test):
        predictions = []

        for item in X_test:
            hidden_init = np.zeros((self.hs_size, 1))

            _, output_probs, _ = self.forward(item, hidden_init)

            ohe_output = np.where(output_probs == np.max(output_probs, axis=1).reshape(-1, 1), 1, 0)

            predictions.append(''.join([self.alphabet[np.argmax(row)] for row in ohe_output]))

        return predictions

    def loss_calc(self, prob, actual):
        return -np.sum(actual * np.log(prob)) / len(prob)

    def ohe(self, word):
        output = []
        for letter in word:
            encoded_symbol = np.zeros(len(self.alphabet))
            encoded_symbol[letter == self.alphabet] = 1
            output.append(encoded_symbol)

        return np.array(output)
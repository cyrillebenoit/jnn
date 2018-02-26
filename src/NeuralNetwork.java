import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class NeuralNetwork {
    private int inputs;
    private int hidden;
    private int outputs;
    private DoubleMatrix[] matrices;
    private DoubleMatrix[] activations;
    private double learningRate;

    /**
     * Main constructor.
     *
     * @param inputs       number of input neurons
     * @param hidden       number of neurons per hidden layer
     * @param hiddenLayers number of hidden layers
     * @param outputs      number of output neurons
     * @throws NeuralNetworkException if one of the parameters is less than 1
     */
    public NeuralNetwork(int inputs, int hidden, int hiddenLayers, int outputs) throws NeuralNetworkException {
        if (inputs < 1 || hidden < 1 || hiddenLayers < 1 || outputs < 1)
            throw new NeuralNetworkException("Lol.");
        this.inputs = inputs;
        this.hidden = hidden;
        this.outputs = outputs;
        this.activations = new DoubleMatrix[hiddenLayers + 2];
        this.learningRate = 1;
        this.matrices = this.initializeRandomMatrix(hiddenLayers + 1);
    }

    /**
     * Learning rate setter.
     *
     * @param learningRate new learning rate value
     */
    public void setLearningRate(double learningRate) {
        if (learningRate > 0)
            this.learningRate = learningRate;
    }

    /**
     * This function returns a random list of inputs.
     *
     * @return double[] random list of inputs.
     */
    public double[] getRandomInputs() {
        return DoubleMatrix.rand(inputs).toArray();
    }

    /**
     * This function is used to get a prediction from a list of inputs
     *
     * @param inputs list of input neurons activations
     * @return list of outputs activations
     * @throws NeuralNetworkException if inputs are not of the right length.
     */
    public double[] guess(double[] inputs) throws NeuralNetworkException {
        DoubleMatrix inputMatrix = new DoubleMatrix(inputs);
        this.checkInput(inputMatrix);
        return guess(inputMatrix).toArray();
    }

    /**
     * This function is used to train the neural network for one single element. The gradient descent is done immediately.
     *
     * @param trainingData TrainingData object
     * @throws NeuralNetworkException if the Training Data is invalid for this neural network.
     */
    public void simpleTrain(TrainingData trainingData) throws NeuralNetworkException {
        checkTrainingData(trainingData);
        DoubleMatrix[] changes = this.train(trainingData.getInputs(), trainingData.getExpected());
        saveGradientDescent(changes);
    }

    /**
     * This function is used to train the neural network for a mini batch of elements.
     * The gradient descent is done for the mean of all individual gradient descent.
     *
     * @param trainingDatum Array of TrainingData object
     * @throws NeuralNetworkException if the Training Data is invalid for this neural network.
     */
    public void batchTrain(TrainingData[] trainingDatum) throws NeuralNetworkException {
        checkTrainingDatum(trainingDatum);
        DoubleMatrix[] batchGradientDescent = this.initializeZeroMatrix(this.matrices.length);
        for (TrainingData trainingData : trainingDatum) {
            DoubleMatrix[] partialGradientDescent = this.train(trainingData.getInputs(), trainingData.getExpected());
            for (int i = 0; i < batchGradientDescent.length; i++) {
                batchGradientDescent[i].addi(partialGradientDescent[i].div(trainingDatum.length));
            }
        }
        saveGradientDescent(batchGradientDescent);
    }

    //TODO stochastic mini batch training

    /**
     * This function returns a clone of the calling object.
     */
    public NeuralNetwork clone() {
        return new NeuralNetwork(this.inputs, this.hidden, this.outputs, this.matrices, this.activations, this.learningRate);
    }

    /**
     * Constructor used by clone function.
     *
     * @param inputs       input neurons
     * @param hidden       hidden neurons
     * @param outputs      output neurons
     * @param matrices     matrices of weights and biases
     * @param activations  activations
     * @param learningRate learning rate
     */
    private NeuralNetwork(int inputs, int hidden, int outputs, DoubleMatrix[] matrices, DoubleMatrix[] activations, double learningRate) {
        this.inputs = inputs;
        this.hidden = hidden;
        this.outputs = outputs;
        this.matrices = matrices.clone();
        this.activations = activations.clone();
        this.learningRate = learningRate;
    }

    /**
     * This function initializes all weights and biases to random values.
     */
    private DoubleMatrix[] initializeRandomMatrix(int length) {
        DoubleMatrix[] res = new DoubleMatrix[length];
        res[0] = DoubleMatrix.rand(this.hidden, this.inputs + 1);
        for (int pointer = 1; pointer < length - 1; pointer++) {
            res[pointer] = DoubleMatrix.rand(this.hidden, this.hidden + 1);
        }
        res[length - 1] = DoubleMatrix.rand(this.outputs, this.hidden + 1);
        return res;
    }

    /**
     * This function initializes all weights and biases to zeros.
     */
    private DoubleMatrix[] initializeZeroMatrix(int length) {
        DoubleMatrix[] res = new DoubleMatrix[length];
        res[0] = DoubleMatrix.zeros(this.hidden, this.inputs + 1);
        for (int pointer = 1; pointer < length - 1; pointer++) {
            res[pointer] = DoubleMatrix.zeros(this.hidden, this.hidden + 1);
        }
        res[length - 1] = DoubleMatrix.zeros(this.outputs, this.hidden + 1);
        return res;
    }

    /**
     * This function is used to get a prediction from a matrix of inputs
     *
     * @param inputs matrix of input neurons activations
     * @return matrix of outputs activations
     */
    private DoubleMatrix guess(DoubleMatrix inputs) {
        this.activations[0] = inputs;
        for (int pointer = 0; pointer < this.matrices.length; pointer++) {
            inputs = DoubleMatrix.concatVertically(inputs, DoubleMatrix.ones(1));
            inputs = this.matrices[pointer].mmul(inputs);
            inputs = MyMatrixFunctions.sigmoid(inputs);
            this.activations[pointer + 1] = inputs;
        }
        return inputs;
    }

    /**
     * This function is used to train the neural network from a matrix of inputs and an expected output.
     *
     * @param inputs matrix of inputs
     * @param answer matrix of expected outputs
     */
    private DoubleMatrix[] train(DoubleMatrix inputs, DoubleMatrix answer) {
        DoubleMatrix guess = this.guess(inputs);
        DoubleMatrix outputErrors = answer.sub(guess);
        DoubleMatrix[] changes = new DoubleMatrix[this.matrices.length];
        // backward propagation
        for (int layer = this.activations.length - 1; layer > 0; layer--) {
            // for each layer (starting from the end)
            DoubleMatrix weights_ = this.matrices[layer - 1];
            inputs = this.activations[layer - 1];
            DoubleMatrix outputs_ = this.activations[layer];
            // calculate gradient for weights and biases nudges
            DoubleMatrix gradientProduct = MyMatrixFunctions.dsigmoid(new DoubleMatrix(outputs_.toArray()));
            DoubleMatrix gradient = gradientProduct.mul(outputErrors);
            DoubleMatrix delta = gradient.mmul(inputs.transpose());
            delta = DoubleMatrix.concatHorizontally(delta, gradient);
            // This is what should change for both the weights and the biases for this layer
            changes[layer - 1] = delta;

            // calculate error of previous layer
            DoubleMatrix input_errors = new DoubleMatrix(inputs.rows);
            DoubleMatrix rowsSums = MatrixFunctions.abs(weights_).rowSums().sub(MatrixFunctions.abs(weights_).getColumn(weights_.columns - 1));
            for (int input_neuron = 0; input_neuron < inputs.length; input_neuron++) {
                double input_error = 0;
                for (int output = 0; output < outputs_.length; output++) {
                    // Distribute the error backwards proportionally to the weights.
                    double proportion = Math.abs(weights_.get(output, input_neuron)) / rowsSums.get(output);
                    input_error += outputErrors.get(output) * proportion;
                }
                // Add this input error to the array of errors in the previous layer
                input_errors.put(input_neuron, input_error);
            }
            // Use input error as output error for next loop
            outputErrors = input_errors;
        }
        for (DoubleMatrix change : changes) {
            change.mul(this.learningRate);
        }
        return changes;
    }

    /**
     * This function is used to add the calculated gradient descent on the weights and biases.
     *
     * @param changes Array of DoubleMatrices.
     */
    private void saveGradientDescent(DoubleMatrix[] changes) {
        for (int i = 0; i < this.matrices.length; i++)
            this.matrices[i].addi(changes[i]);
    }

    /**
     * This function is used to check if an input matrix is valid.
     *
     * @param inputs input matrix.
     * @throws NeuralNetworkException If the inputs are wrongly dimensioned.
     */
    private void checkInput(DoubleMatrix inputs) throws NeuralNetworkException {
        if (inputs.length != this.inputs || !inputs.isColumnVector())
            throw new NeuralNetworkException("Inputs invalid.");
    }

    /**
     * This function is used to check if an expected output matrix is valid.
     *
     * @param outputs expected output matrix.
     * @throws NeuralNetworkException If the expected outputs are wrongly dimensioned.
     */
    private void checkOutput(DoubleMatrix outputs) throws NeuralNetworkException {
        if (outputs.length != this.outputs || !outputs.isColumnVector())
            throw new NeuralNetworkException("Outputs invalid.");
    }

    /**
     * This function is used to check if a training data object is correct for the neural network.
     *
     * @param trainingData Training Data.
     * @throws NeuralNetworkException If the Training Data is invalid.
     */
    private void checkTrainingData(TrainingData trainingData) throws NeuralNetworkException {
        try {
            this.checkInput(trainingData.getInputs());
            this.checkOutput(trainingData.getExpected());
        } catch (NeuralNetworkException e) {
            throw new NeuralNetworkException("Training data invalid.");
        }
    }

    /**
     * This function is used to check if an array of training data objects is correct for the neural network.
     *
     * @param trainingData Training Datum.
     * @throws NeuralNetworkException If the Training Datum is invalid.
     */
    private void checkTrainingDatum(TrainingData[] trainingData) throws NeuralNetworkException {
        for (TrainingData td : trainingData)
            this.checkTrainingData(td);
    }
}

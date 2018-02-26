import org.jblas.DoubleMatrix;

public class TrainingData {
    private DoubleMatrix inputs;
    private DoubleMatrix expected;

    public TrainingData(double[] inputs, double[] expected) {
        this.inputs = new DoubleMatrix(inputs);
        this.expected = new DoubleMatrix(expected);
    }

    DoubleMatrix getInputs() {
        return inputs;
    }

    DoubleMatrix getExpected() {
        return expected;
    }
}

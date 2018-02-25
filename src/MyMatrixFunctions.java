import org.jblas.DoubleMatrix;

class MyMatrixFunctions {

    /**
     * Applies the <i>sigmoid</i> function element wise on this
     * matrix. Note that this is an in-place operation.
     *
     * @return this matrix
     */
    static DoubleMatrix sigmoid(DoubleMatrix x) {
        for (int i = 0; i < x.length; i++)
            x.put(i, 1.0 / (1 + Math.exp(-x.get(i))));
        return x;
    }

    /**
     * Applies the <i>dsigmoid</i> function element wise on this
     * matrix. Note that this is an in-place operation.
     *
     * @return this matrix
     */
    static DoubleMatrix dsigmoid(DoubleMatrix x) {
        for (int i = 0; i < x.length; i++)
            x.put(i, x.get(i) * (1 - x.get(i)));
        return x;
    }
}

// To calculate Q,K,V matrices
// we take X, which is the input matrix of size nxd, where n is number of tokens
// and d is number of dimensions in the embedding
// and do Q = XWq, K = XWk, V = XWv
//
// Attention score = Q x Transpose of K
// Scale it down by dividing it by sqrt(dk), dk is dimension of key vectors
// Attention weights = softmax(the above)
// Output = Attention weights x V
//
// For multihead attention, we split input sequence into smaller segemnets,
// process each separately, then concatenate all the weight matrices together
//
// Afterwards, we process through FFN
// FFN(x) = ReLU(xW1 + b1)W2 + b2
// where x is the input of the activation function
// W1 and W2 are matrices, and b1 and b2 are bias vectors

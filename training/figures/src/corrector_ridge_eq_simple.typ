// The ridge corrector in one line: weighted ridge regression from standardised observation features
// to the nudge (ttt.correctors.ridge; full version in corrector_ridge_eq.typ).
#set page(width: 12cm, height: auto, margin: 14pt, fill: white)
#set text(font: "New Computer Modern", size: 11pt, fill: black)

$ hat(W), hat(b) = arg min_(W, b) sum_(i=1)^N w_i norm(tilde(x)_i^T W + b^T - y_i^T)^2 + lambda norm(W)_F^2 $

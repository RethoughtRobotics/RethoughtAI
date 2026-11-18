// Mock-up of the "Guaranteeing the Base Policy" slide (16:9): the kernel corrector figure without its
// caption (corrector_kernel.typ) and the guarantee in one line, from first principles.
#set page(width: 25.4cm, height: auto, margin: (x: 1.6cm, y: 1.0cm), fill: white)
#set text(font: "New Computer Modern", size: 16pt, fill: black)

#text(size: 28pt)[Guaranteeing the Base Policy]
#v(0.3cm)

Store each correction. Nudge only when the robot is *close* to one; otherwise the weight is exactly 0.
#v(0.2cm)
#align(center, image("assets/corrector_kernel_nocaption.png", height: 6.2cm))
#v(0.2cm)
*Guarantee:* far from every correction the nudge is exactly 0: the base policy runs unchanged.

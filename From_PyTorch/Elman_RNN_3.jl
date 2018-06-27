#=
Added another middle layer to Elman_RNN.jl - gives better results
=#
using Flux
using Flux.Tracker
using Plots

srand(1)

input_size, h1_size,h2_size, output_size = 8, 7, 6, 1
epochs = 500
seq_length = 20
lr = 0.05

data_time_steps = linspace(2,10, seq_length+1)
data = sin.(data_time_steps)

x = data[1:end-1]
y = data[2:end]

w1 = param(randn(input_size,  h1_size))
w2 = param(randn(h1_size, h2_size))
w3 = param(randn(h2_size, output_size)) 

function forward(input, context_state, W1, W2, W3)
   #xh = cat(2,input, context_state) 
   #  Due to a Flux bug you have to do:
   xh = cat(2, Tracker.collect(input), context_state) 
   context_state = tanh.(xh*W1)
   l2 = tanh.(context_state*W2)
   out = l2*W3
   return out, context_state
end
lrs = []
function train(lri=lr) 
   for i in 1:epochs
      lr = lri
      total_loss = 0
      context_state = param(zeros(1,h1_size))
      for j in 1:length(x)
        input = x[j]
        target= y[j]
        pred, context_state = forward(input, context_state, w1, w2, w3)
        loss = sum((pred .- target).^2)/2
        total_loss += loss
        back!(loss)
        w1.data .-= lr.*w1.grad
        w2.data .-= lr.*w2.grad
        w3.data .-= lr.*w3.grad

        lr *= 0.95
 #lr = lri/(1+(j*i)/20)
        append!(lrs, lr)
        
        w1.grad .= 0.0
        w2.grad .= 0.0 
        w3.grad .= 0.0 
        context_state = param(context_state.data)
      end
      if(i % 10 == 0)
        println("Epoch: $i  loss: $total_loss")
      end
   end
end

train()

context_state = param(zeros(1,h1_size))
predictions = []

println("... running forward test ... ")
for i in 1:length(x)
  input = x[i]
  pred, context_state = forward(input, context_state, w1, w2, w3)
  append!(predictions, pred.data)
end
println("... plot ... ")
Plots.backend(:gr)
plot(lrs)
   """
scatter(data_time_steps[1:end-1], x, label="actual")
scatter!(data_time_steps[2:end],predictions, label="predicted")
 #png("scatter.png")
"""


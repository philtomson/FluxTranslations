#=
GPU version
Translated From:
Introduction to Recurrent Neural Networks in Pytorch
https://www.cpuheater.com/deep-learning/introduction-to-recurrent-neural-networks-in-pytorch/
GitHub: https://github.com/cpuheater/pytorch_examples
=#
println("START")
@time using Flux
println("used Flux")
@time using Flux.Tracker
println("used Flux.Tracker")
 #using Plots
@time using CuArrays
println("used CuArrays")

srand(1)

input_size, hidden_size, output_size = 7, 6, 1
epochs = 200
seq_length = 20
lr = 0.1

println("1")
data_time_steps = (linspace(2,10, seq_length+1))
data = (sin.(data_time_steps))

println("2")
x = gpu(data[1:end-1])
y = gpu(data[2:end])
println("3")

w1 = (param(gpu(randn(input_size,  hidden_size))))
w2 = (param(gpu(randn(hidden_size, output_size))))
println("4")

function forward(input, context_state, W1, W2)
   #xh = cat(2,input, context_state) 
   #  Due to a Flux bug you have to do:
   xh = cu(cat(2, Tracker.collect(input), context_state))
   context_state = CUDAnative.tanh.(xh*W1)
   out = context_state*W2
   return out, context_state
end

function train() 
   for i in 1:epochs
      total_loss = 0
      context_state = gpu(zeros(1,hidden_size))
      for j in 1:length(x)
        input = x[j]
        target= y[j]
        pred, context_state = forward(input, context_state, w1, w2)
        loss = sum((pred .- target).^2)/2
        total_loss += loss
        back!(loss)
        w1.data .-= lr.*w1.grad
        w2.data .-= lr.*w2.grad

        w1.grad .= 0.0
        w2.grad .= 0.0 
        context_state = gpu(context_state.data) 
      end
      if(i % 10 == 0)
        println("Epoch: $i  loss: $total_loss")
      end
   end
end

function predict(x,w1,w2)
   context_state = gpu(zeros(1,hidden_size))
   predictions = []
   for i in 1:length(x)
     input = x[i]
     pred, context_state = forward(input, context_state, w1, w2)
     append!(predictions, pred.data)
   end
   predictions
end

println("train")
@time train()
println("predict")
@time predictions = predict(x,w1,w2)

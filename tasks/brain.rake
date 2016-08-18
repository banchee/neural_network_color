namespace :brain do
  require 'pry'
  require './network/network'

  desc "Train neural network"
  task :train, [:data_set, :neural_network_layout] do |_, args|
    brain = Network.new([4, 1, [4], 4])

    # t = brain.layers.map.with_index do |layer, index_i|
    #   layer.neurons.map.with_index do |neuron, index_j|
    #     [index_i,[index_j, neuron.synapses.map(&:weight)]]
    #   end
    # end

    # puts "Before"
    # puts t
    # puts "End"

    # binding.pry

    brain.train([[[0,0,0,0],0],
                 [[0,0,0,1],1],
                 [[0,0,1,0],2],
                 [[0,0,1,1],3]])



    # t = brain.layers.map.with_index do |layer, index_i|
    #   layer.neurons.map.with_index do |neuron, index_j|
    #     [index_i,[index_j, neuron.synapses.map(&:weight)]]
    #   end
    # end

    # puts "After"
    # puts t
    # puts "End"
    # binding.pry

    [[[0,0,0,0],0],
     [[0,0,0,1],1],
     [[0,0,1,0],2],
     [[0,0,1,1],3]].each do |row|

       answer = brain.run(row[0])

       puts 'got : ' + answer.max.to_s + ' expecpted : ' + row[1].to_s


    end



  end

end

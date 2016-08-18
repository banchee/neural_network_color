class Synapse

  attr_accessor :weight, :input, :new_weight

  def initialize(weight=nil)
    @weight = weight
  end

  def store_new_weight(difference)
    @new_weight = weight + difference
  end

  def firing_rate(input)
    input * weight
  end

  def update!
    @weight = new_weight unless new_weight.nil?
    clear_new_weight
    self
  end

  def clear_new_weight
    @new_weight = nil
  end

end

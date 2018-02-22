#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <Eigen/Dense>

namespace ops {
    
    static boost::mt19937 rng;

    struct Initializer {
        virtual Eigen::Ref<const Eigen::MatrixXd> operator()() = 0;
    };
    
    template<std::size_t R, std::size_t C>
    struct ZeroInitializer : public Initializer {
        Eigen::Ref<const Eigen::MatrixXd> operator()() override {
            return Eigen::MatrixXd::Zero(R,C);
        }
    };
 
 
    template<typename ValueType, std::size_t R, std::size_t C>
    struct UniformInitializer : public Initializer {
        UniformInitializer(const double lower, const double upper)
        : m_lower(lower), m_upper(upper), unifd(boost::random::uniform_real_distribution<>(lower,upper)) {}
        
        Eigen::Ref<const Eigen::MatrixXd> operator()() override {
            try {
                
                return Eigen::MatrixXd::Zero(R,C).unaryExpr([&](int element){ return unifd(rng); });
            }
            catch (const std::exception& boost_exception) {
                throw; // boost can handle this better
            }
        }
        
    private:
        double m_lower;
        double m_upper;
        boost::random::uniform_real_distribution<> unifd;
    };
     

    template<std::size_t R, std::size_t C>
    struct NormalInitializer : public Initializer {
        NormalInitializer(const double mean, const double var)
        : m_mean(mean), m_var(var) {}
        Eigen::Ref<const Eigen::MatrixXd> operator()()  {
            try {
                boost::normal_distribution<> gaussd(m_mean, m_var);
                boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > gaussvars(rng, gaussd);
                return Eigen::MatrixXd::Zero(R,C).unaryExpr([&](double element){ return gaussvars(); });
            }
            catch (const std::exception& boost_exception) {
                throw; // boost can handle this better
            }
        }
    private:
        double m_mean;
        double m_var;
    };
    

    template<std::size_t R, std::size_t C>
    class Matrix {
    public:
        
        Matrix() = default;
        
        Matrix(const Eigen::MatrixXd&& mat) {
            if (mat.rows() != R && mat.cols() != C) {
                throw std::exception();
            }
            m_mat = mat;
        }
        
        Eigen::MatrixXd& mat()  {
            return m_mat;
        }
        
        std::size_t nrows() const {
            return m_nrows;
        }
        
        std::size_t ncols() const {
            return m_ncols;
        }
        
        void normalize() {
            
        }
        
        template<typename InitializerFunc>
        void initialize(InitializerFunc initializer) {
            if (m_initialized)
                throw std::runtime_error("Error: calling initialize on an initiailized matrix.\n");
            m_mat = initializer();
        }
        
    private:
        std::size_t m_nrows = R;
        std::size_t m_ncols = C;
        Eigen::MatrixXd m_mat;
        bool m_initialized = false;
    };
    
    
}

int main() {

    using namespace ops;
    auto a = Matrix<2,2>();
    UniformInitializer<double, 4, 4> m(1, 1.1);
    a.initialize(m);
    auto b = a.mat();
    
    Eigen::Matrix3d x;
    std::cout << x << std::endl;
}



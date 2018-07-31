inline std::tuple<cv::Mat, cv::Mat, float> ulsee_fitting::weak_perspective( cv::Mat p2ts,
                                                                            cv::Mat p3ts)
{

//    return calcPose(p2ts,p3ts);

    assert(p2ts.rows/2 == p3ts.rows/3);
    int num = p2ts.rows/2;
    cv::Mat mean2d_(2, 1, CV_32FC1, cv::Scalar(0)) ;
    cv::Mat mean3d_(3, 1, CV_32FC1, cv::Scalar(0));

    p2ts = p2ts.reshape(0,num);
    p2ts = p2ts.t();

    p3ts = p3ts.reshape(0,num);
    p3ts = p3ts.t();

//    cv::reduce(p2ts,mean2d_,1,cv::REDUCE_AVG);
//    cv::reduce(p3ts,mean3d_,1,cv::REDUCE_AVG);
//    float x_bar = mean2d_.at<float>(0,0), y_bar = mean2d_.at<float>(1,0);
//    float X_bar = mean3d_.at<float>(0,0), Y_bar = mean3d_.at<float>(1,0), Z_bar = mean3d_.at<float>(2,0);
    float x_bar = 0, y_bar = 0, X_bar = 0, Y_bar = 0, Z_bar = 0;

    //*************centralization/normalization
    for (int i = 0; i < num; i++)
    {
        x_bar += p2ts.at<float>(0, i);
        y_bar += p2ts.at<float>(1, i);
        X_bar += p3ts.at<float>(0, i);
        Y_bar += p3ts.at<float>(1, i);
        Z_bar += p3ts.at<float>(2, i);
    }

    x_bar /= num;
    y_bar /= num;
    X_bar /= num;
    Y_bar /= num;
    Z_bar /= num;



    cv::Mat p3d = cv::Mat::zeros(2 * num, 6, CV_32FC1);
    cv::Mat pts = cv::Mat::zeros(2 * num, 1, CV_32FC1);

    for (int i = 0; i < num; i++)
    {
        pts.at<float>(2 * i, 0) = p2ts.at<float>(0, i) - x_bar;
        pts.at<float>(2 * i + 1, 0) = p2ts.at<float>(1, i) - y_bar;

        float distance_X = p3ts.at<float>(0, i) - X_bar;
        float distance_Y = p3ts.at<float>(1, i) - Y_bar;
        float distance_Z = p3ts.at<float>(2, i) - Z_bar;

        p3d.at<float>(2 * i, 0) = distance_X;
        p3d.at<float>(2 * i, 1) = distance_Y;
        p3d.at<float>(2 * i, 2) = distance_Z;

        p3d.at<float>(2 * i + 1, 3) = distance_X;
        p3d.at<float>(2 * i + 1, 4) = distance_Y;
        p3d.at<float>(2 * i + 1, 5) = distance_Z;
    }
    /////////////////////******************************************************/////////////////////

    // use p3d, p2ts;;;;;;;
    cv::Mat c_s;




    cv::Mat left = p3d.t() * p3d;
    cv::Mat right = p3d.t() * pts;
    cv::solve(left, right, c_s, cv::DECOMP_SVD);

    cv::Mat alpha(3, 1, CV_32FC1), gamma(3, 1, CV_32FC1);
    //for(int i = 0 ; i < c_s.rows; i++){
    // std:: cout << c_s.at<double>(i, 0) << std::endl;
    //}
    c_s.rowRange(0, 3).copyTo(alpha);
    c_s.rowRange(3, 6).copyTo(gamma);


    //orthonormalization
    float d1 = std::sqrt((alpha.dot(alpha)) * (gamma.dot(gamma)) - (alpha.dot(gamma)) * (alpha.dot(gamma)));
    float d2 = (alpha.dot(alpha)) * (gamma.dot(gamma)) + cv::norm(alpha) * cv::norm(gamma) * d1 -
               (alpha.dot(gamma)) * (alpha.dot(gamma));

    float v1 = (cv::norm(alpha) + cv::norm(gamma)) / (2 * cv::norm(alpha)) +
               (cv::norm(gamma) * (alpha.dot(gamma)) * (alpha.dot(gamma))) / (2 * cv::norm(alpha) * d2);
    float v2 = (alpha.dot(gamma)) / (2 * d1);
    //debuging
    //std:: cout << v1 << "   " << v2 << std::endl;

    cv::Mat alpha_bar = alpha * v1 - gamma * v2;

    float v3 = (alpha.dot(gamma)) / (2 * d1);

    float v4 = (cv::norm(alpha) + cv::norm(gamma)) / (2 * cv::norm(gamma)) +
               (cv::norm(alpha) * (alpha.dot(gamma)) * (alpha.dot(gamma))) / (2 * cv::norm(gamma) * d2);
    //debuging
    //std:: cout << v3 << "   " << v4 << std:: endl;

    cv::Mat gamma_bar = alpha * v3 - gamma * v4;

    /* std::cout << cv::norm(alpha_bar) << "    " << cv::norm(gamma_bar) << "    " <<
                  alpha_bar.dot(gamma_bar) << std::endl;*/

    float scale = (cv::norm(alpha_bar) + cv::norm(gamma_bar))/2.0f;
    alpha_bar = alpha_bar.t() / scale;
    gamma_bar = -gamma_bar.t() / scale;



    cv::Mat R(3, 3, CV_32FC1);

    alpha_bar.copyTo(R.rowRange(0, 1));
    gamma_bar.copyTo(R.rowRange(1, 2));

    cv::Mat theta_bar = R.rowRange(0, 1).cross(R.rowRange(1, 2));
    theta_bar.copyTo(R.rowRange(2, 3));

    //correct yaw rotation
//    R.convertTo(R, CV_64FC1, 1.0, 0);
//    cv::Vec3d angle = RotationMatrixToEulerAngles(R);
//    R = eulerAnglesToRotationMatrix(angle);
//    R.convertTo(R, CV_32FC1, 1.0, 0);



    cv::Mat mean2d = (cv::Mat_<float>(3, 1) << x_bar, y_bar, 0.0);
    cv::Mat mean3d = (cv::Mat_<float>(3, 1) << X_bar, Y_bar, Z_bar);
    cv::Mat T = mean2d - (R * scale * mean3d);

    std::cout <<"Rt*R = " <<R.t()*R<<std::endl;

    return std::make_tuple(R, T, scale);
}
